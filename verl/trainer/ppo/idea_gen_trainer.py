# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import uuid
from collections import defaultdict
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    compute_response_mask,
    compute_advantage,
    apply_kl_penalty,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


def parse_model_output(response_str: str) -> Tuple[str, str]:
    cot = ""
    method = ""

    think_end_match = re.search(r"</think>", response_str, re.IGNORECASE)
    if think_end_match:
        cot_part = response_str[:think_end_match.start()]
        think_start_match = re.search(r"<think>", cot_part, re.IGNORECASE)
        cot = cot_part[think_start_match.end():].strip() if think_start_match else cot_part.strip()

        method_part = response_str[think_end_match.end():]
        method_header_match = re.search(r"##\s*Method", method_part, re.IGNORECASE)
        method = method_part[method_header_match.end():].strip() if method_header_match else method_part.strip()
    else:
        method_header_match = re.search(r"##\s*Method", response_str, re.IGNORECASE)
        if method_header_match:
            cot = response_str[:method_header_match.start()].strip()
            method = response_str[method_header_match.end():].strip()
        else:
            method = response_str.strip()

    return cot, method


class IdeaGenTrainer(RayPPOTrainer):

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager: ResourcePoolManager,
        **kwargs
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            **kwargs
        )


        reward_kwargs = self.config.reward_model.get('reward_kwargs', {})


        self.use_entropy_reward = self.config.reward_model.get('reward_manager', 'naive') == 'entropy'


        self.global_step = 0


        self.use_eaig = reward_kwargs.get('use_eaig', True)
        self.eaig_weight = reward_kwargs.get('eaig_weight', 0.4)
        self.eaig_top_p_ratio = reward_kwargs.get('eaig_top_p_ratio', 0.20)


        self.use_eaig_shaping = reward_kwargs.get('use_eaig_shaping', True)


        self.eaig_shape_thresholds = reward_kwargs.get('eaig_shape_thresholds', [1.0, 1.5, 2.0])
        self.eaig_shape_rewards = reward_kwargs.get('eaig_shape_rewards', [0.0, 0.5, 0.8, 1.0])


        self.use_length_anchor = reward_kwargs.get('use_length_anchor', True)
        self.target_cot_length = reward_kwargs.get('target_cot_length', None)
        self.length_anchor_lambda = reward_kwargs.get('length_anchor_lambda', 0.5)
        self._length_anchor_initialized = False


        self.semantic_weight = reward_kwargs.get('semantic_weight', 0.6)
        self.use_semantic_shaping = reward_kwargs.get('use_semantic_shaping', True)

        self.semantic_shape_thresholds = [0.01, 0.05, 0.1]
        self.semantic_shape_rewards = [0.0, 0.5, 0.8, 1.0]


        self.use_semantic_similarity = reward_kwargs.get('use_semantic_similarity', True)
        self.embedding_api_url = reward_kwargs.get('embedding_api_url', 'http://localhost:30015/v1/embeddings')


        self.use_overview_only = reward_kwargs.get('use_overview_only', True)
        self.use_contrastive_gain = reward_kwargs.get('use_contrastive_gain', True)


        self.use_format_penalty = reward_kwargs.get('use_format_penalty', True)


    def _load_embedding_model(self):
        if not self.use_semantic_similarity:
            return

        if hasattr(self, '_embedding_api_verified') and self._embedding_api_verified:
            return

        try:
            import requests

            is_openai_format = '/v1/' in self.embedding_api_url
            self._embedding_api_openai_format = is_openai_format


            if is_openai_format:

                test_payload = {"input": ["test"], "model": "default"}
            else:

                test_payload = {"inputs": ["test"]}

            response = requests.post(
                self.embedding_api_url,
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                self._embedding_api_verified = True
            else:
                raise RuntimeError(
                    f"Embedding API verification failed at {self.embedding_api_url}: "
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )

        except Exception as exc:
            raise RuntimeError(
                "Semantic similarity reward is enabled, but the embedding API could not be verified. "
                "Start the embedding service or set reward_model.reward_kwargs.use_semantic_similarity=false."
            ) from exc

    def _get_embeddings_from_api(self, texts: list) -> np.ndarray:
        import requests

        try:

            is_openai_format = getattr(self, '_embedding_api_openai_format', '/v1/' in self.embedding_api_url)

            if is_openai_format:

                payload = {"input": texts, "model": "default"}
            else:

                payload = {"inputs": texts}

            response = requests.post(
                self.embedding_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )

            response.raise_for_status()
            result = response.json()

            if is_openai_format:
                embeddings = [item["embedding"] for item in sorted(result["data"], key=lambda x: x["index"])]
            else:
                embeddings = result

            embeddings = np.array(embeddings, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
                raise ValueError(f"Expected embedding shape ({len(texts)}, dim), got {embeddings.shape}.")
            return embeddings

        except Exception as exc:
            raise RuntimeError(
                f"Embedding API request failed at {self.embedding_api_url}; semantic reward cannot be computed."
            ) from exc

    def _compute_semantic_similarity(self, pred_texts: list, gt_texts: list) -> torch.Tensor:
        if not self.use_semantic_similarity:
            return torch.zeros(len(pred_texts))

        batch_size = len(pred_texts)
        similarities = torch.zeros(batch_size)

        valid_indices = []
        valid_preds = []
        valid_gts = []

        for i, (pred, gt) in enumerate(zip(pred_texts, gt_texts)):
            pred_clean = pred.strip() if pred else ""
            gt_clean = gt.strip() if gt else ""

            if pred_clean and gt_clean:
                valid_indices.append(i)
                valid_preds.append(pred_clean)
                valid_gts.append(gt_clean)

        if not valid_indices:
            return similarities

        pred_embeddings = self._get_embeddings_from_api(valid_preds)
        gt_embeddings = self._get_embeddings_from_api(valid_gts)

        pred_tensor = torch.tensor(pred_embeddings, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_embeddings, dtype=torch.float32)

        pred_norm = torch.nn.functional.normalize(pred_tensor, p=2, dim=1)
        gt_norm = torch.nn.functional.normalize(gt_tensor, p=2, dim=1)
        cos_sim = (pred_norm * gt_norm).sum(dim=1)
        cos_sim_normalized = (cos_sim + 1) / 2

        for idx, sim in zip(valid_indices, cos_sim_normalized):
            similarities[idx] = sim.item()

        return similarities

    def _extract_method_parts(self, method_text: str) -> tuple:
        if not method_text or not method_text.strip():
            return "", ""

        text = method_text.strip()


        if text.lower().startswith("## method"):
            text = text[len("## method"):].strip()


        split_pos = text.find("### ")

        if split_pos == -1:

            for marker in ["1.", "1)"]:
                pos = text.find(marker)
                if pos != -1:
                    split_pos = pos
                    break

        if split_pos == -1:

            overview = text.strip()
            details = ""
        else:

            overview = text[:split_pos].strip()
            details = text[split_pos:].strip()

        return overview, details

    def _check_cot_format_violation(self, cot_text: str) -> bool:
        if not cot_text:
            return False


        header_pattern = r'^#{2,}\s+'
        if re.search(header_pattern, cot_text, re.MULTILINE):
            return True


        method_indicators = [
            r'^##\s+Method',
            r'^###\s+',
            r'^\*\*Step\s+\d+',
        ]

        for pattern in method_indicators:
            if re.search(pattern, cot_text, re.MULTILINE | re.IGNORECASE):
                return True

        return False

    def _compute_eaig_reward(
        self,
        base_log_p: torch.Tensor,
        cot_log_p: torch.Tensor,
        valid_mask: torch.Tensor,
        base_entropy: Optional[torch.Tensor] = None,
    ) -> tuple:
        batch_size = base_log_p.shape[0]
        device = base_log_p.device


        if base_entropy is not None:
            uncertainty = base_entropy * valid_mask
            use_true_entropy = True
        else:
            uncertainty = (-base_log_p) * valid_mask
            use_true_entropy = False


        eaig_rewards = torch.zeros(batch_size, device=device)
        mask_counts = torch.zeros(batch_size, device=device)
        total_token_counts = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            valid_uncertainty = uncertainty[i][valid_mask[i] > 0]
            if len(valid_uncertainty) == 0:
                continue


            total_token_counts[i] = len(valid_uncertainty)


            threshold = torch.quantile(valid_uncertainty, 1 - self.eaig_top_p_ratio)


            mask = ((uncertainty[i] > threshold) & (valid_mask[i] > 0)).float()


            ig = cot_log_p[i] - base_log_p[i]
            masked_ig = ig * mask


            mask_sum = mask.sum()
            if mask_sum > 0:
                eaig_rewards[i] = masked_ig.sum() / mask_sum
                mask_counts[i] = mask_sum


        info = {
            "eaig_mean": eaig_rewards.mean().item(),
            "eaig_std": eaig_rewards.std().item(),
            "mask_count_mean": mask_counts.mean().item(),
            "total_token_count_mean": total_token_counts.mean().item(),
            "use_true_entropy": use_true_entropy,
        }

        return eaig_rewards, info

    def _shape_eaig_reward(self, eaig: torch.Tensor) -> torch.Tensor:
        shaped = torch.zeros_like(eaig)

        thresholds = self.eaig_shape_thresholds
        rewards = self.eaig_shape_rewards


        shaped = torch.where(eaig < thresholds[0],
                            torch.tensor(rewards[0], device=eaig.device), shaped)

        shaped = torch.where((eaig >= thresholds[0]) & (eaig < thresholds[1]),
                            torch.tensor(rewards[1], device=eaig.device), shaped)

        shaped = torch.where((eaig >= thresholds[1]) & (eaig < thresholds[2]),
                            torch.tensor(rewards[2], device=eaig.device), shaped)

        shaped = torch.where(eaig >= thresholds[2],
                            torch.tensor(rewards[3], device=eaig.device), shaped)

        return shaped

    def _shape_semantic_reward(self, r_gain: torch.Tensor) -> torch.Tensor:
        shaped = torch.zeros_like(r_gain)

        thresholds = self.semantic_shape_thresholds
        rewards = self.semantic_shape_rewards


        shaped = torch.where(r_gain < thresholds[0],
                            torch.tensor(rewards[0], device=r_gain.device), shaped)
        shaped = torch.where((r_gain >= thresholds[0]) & (r_gain < thresholds[1]),
                            torch.tensor(rewards[1], device=r_gain.device), shaped)
        shaped = torch.where((r_gain >= thresholds[1]) & (r_gain < thresholds[2]),
                            torch.tensor(rewards[2], device=r_gain.device), shaped)
        shaped = torch.where(r_gain >= thresholds[2],
                            torch.tensor(rewards[3], device=r_gain.device), shaped)

        return shaped

    def _compute_pairwise_similarity(self, texts_a: list, texts_b: list) -> np.ndarray:
        if not texts_a or not texts_b or len(texts_a) != len(texts_b):
            return np.zeros(len(texts_a) if texts_a else 0)

        emb_a = self._get_embeddings_from_api(texts_a)
        emb_b = self._get_embeddings_from_api(texts_b)

        emb_a_norm = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
        emb_b_norm = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)

        cos_sim = (emb_a_norm * emb_b_norm).sum(axis=1)
        return cos_sim

    def _compute_contrastive_semantic_gain(
        self,
        gen_overviews: list,
        gt_overviews: list,
        prompts: list,
    ) -> tuple:
        batch_size = len(gen_overviews)
        r_gain = torch.zeros(batch_size)

        info = {
            "s_base_mean": 0.0,
            "s_model_mean": 0.0,
            "r_gain_mean": 0.0,
        }

        valid_indices = []
        valid_gen_overviews = []
        valid_gt_overviews = []
        valid_prompts = []

        for i in range(batch_size):
            gen_ov = gen_overviews[i].strip() if gen_overviews[i] else ""
            gt_ov = gt_overviews[i].strip() if gt_overviews[i] else ""
            prompt = prompts[i].strip() if prompts[i] else ""

            if gen_ov and gt_ov and prompt:
                valid_indices.append(i)
                valid_gen_overviews.append(gen_ov)
                valid_gt_overviews.append(gt_ov)
                valid_prompts.append(prompt)

        if not valid_indices:
            return r_gain, info

        s_base = self._compute_pairwise_similarity(valid_prompts, valid_gt_overviews)
        s_model = self._compute_pairwise_similarity(valid_gen_overviews, valid_gt_overviews)

        if self.use_contrastive_gain:
            gain = np.maximum(0, s_model - s_base)
        else:
            gain = (s_model + 1) / 2

        for j, idx in enumerate(valid_indices):
            r_gain[idx] = gain[j]

        info["s_base_mean"] = float(s_base.mean())
        info["s_model_mean"] = float(s_model.mean())
        info["r_gain_mean"] = float(gain.mean())

        return r_gain, info

    def _convert_to_python_list(self, obj):
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            if obj.shape == ():
                return self._convert_to_python_list(obj.item())
            return [self._convert_to_python_list(item) for item in obj.tolist()]
        if isinstance(obj, list):
            return [self._convert_to_python_list(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._convert_to_python_list(v) for k, v in obj.items()}
        return obj

    def _get_non_tensor_item(self, batch: DataProto, key: str, idx: int, default=None):
        values = batch.non_tensor_batch.get(key, None)
        if values is None or idx >= len(values):
            return default
        value = self._convert_to_python_list(values[idx])
        return default if value is None else value

    def _get_ground_truth(self, batch: DataProto, idx: int) -> dict:
        rm_data = self._get_non_tensor_item(batch, "reward_model", idx, {})
        if not isinstance(rm_data, dict):
            return {}
        ground_truth = self._convert_to_python_list(rm_data.get("ground_truth", {}))
        return ground_truth if isinstance(ground_truth, dict) else {}

    def _get_semantic_baseline_text(self, batch: DataProto, idx: int, fallback_prompt: str) -> str:
        extra_info = self._get_non_tensor_item(batch, "extra_info", idx, {})
        ground_truth = self._get_ground_truth(batch, idx)
        if isinstance(extra_info, dict):
            context = extra_info.get("context_raw", "") or ""
            motivation = extra_info.get("motivation_raw", "") or ground_truth.get("motivation_gt", "") or ""
            baseline = "\n\n".join(part.strip() for part in [context, motivation] if part and part.strip())
            if baseline:
                return baseline
        return fallback_prompt

    def _prepare_fwd_batch(self, batch: DataProto) -> Tuple[Optional[DataProto], Optional[torch.Tensor]]:
        try:
            batch_size = len(batch)


            original_input_ids = batch.batch["input_ids"]
            original_attention_mask = batch.batch["attention_mask"]
            original_position_ids = batch.batch["position_ids"]
            original_responses = batch.batch["responses"]
            original_prompts = batch.batch["prompts"]

            prompt_length = original_prompts.shape[-1]
            response_length = original_responses.shape[-1]
            device = original_input_ids.device


            new_input_ids = original_input_ids.clone()
            new_attention_mask = original_attention_mask.clone()
            new_position_ids = original_position_ids.clone()
            new_responses = torch.full_like(original_responses, self.tokenizer.pad_token_id or 0)
            method_gt_mask = torch.zeros((batch_size, response_length), dtype=torch.long, device=device)

            valid_samples = 0

            for i in range(batch_size):


                valid_response_mask = original_attention_mask[i, prompt_length:]
                valid_response_length = int(valid_response_mask.sum().item())
                valid_response_ids = original_responses[i, :valid_response_length]
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)


                cot_text, _ = parse_model_output(response_str)

                ground_truth = self._get_ground_truth(batch, i)
                method_gt = ground_truth.get("method_gt", "")

                if not method_gt or not cot_text:

                    valid_samples += 1
                    continue


                try:
                    cot_ids = self.tokenizer.encode(cot_text, add_special_tokens=False, return_tensors="pt")[0]
                    method_gt_ids = self.tokenizer.encode(method_gt, add_special_tokens=False, return_tensors="pt")[0]
                except Exception:
                    valid_samples += 1
                    continue


                combined_ids = torch.cat([cot_ids, method_gt_ids])
                cot_len = len(cot_ids)
                method_gt_len = len(method_gt_ids)
                combined_len = len(combined_ids)

                if combined_len > response_length:

                    if method_gt_len >= response_length:

                        combined_ids = method_gt_ids[:response_length]
                        cot_len = 0
                        method_gt_len = response_length
                    else:

                        available_for_cot = response_length - method_gt_len
                        cot_ids = cot_ids[-available_for_cot:]
                        combined_ids = torch.cat([cot_ids, method_gt_ids])
                        cot_len = len(cot_ids)
                    combined_len = len(combined_ids)


                pad_len = response_length - combined_len
                if pad_len > 0:
                    padded_response = torch.cat([
                        combined_ids,
                        torch.full((pad_len,), self.tokenizer.pad_token_id or 0, dtype=torch.long),
                    ])
                    response_mask = torch.cat([
                        torch.ones(combined_len, dtype=torch.long),
                        torch.zeros(pad_len, dtype=torch.long),
                    ])

                    method_mask = torch.cat([
                        torch.zeros(cot_len, dtype=torch.long),
                        torch.ones(method_gt_len, dtype=torch.long),
                        torch.zeros(pad_len, dtype=torch.long),
                    ])
                else:
                    padded_response = combined_ids
                    response_mask = torch.ones(response_length, dtype=torch.long)
                    method_mask = torch.cat([
                        torch.zeros(cot_len, dtype=torch.long),
                        torch.ones(method_gt_len, dtype=torch.long)
                    ])


                new_responses[i] = padded_response.to(device)


                new_input_ids[i, prompt_length:] = padded_response.to(device)


                new_attention_mask[i, prompt_length:] = response_mask.to(device)


                full_mask = new_attention_mask[i]
                new_pos = (full_mask.cumsum(dim=-1) - 1).clamp(min=0)
                new_position_ids[i] = new_pos


                method_gt_mask[i] = method_mask.to(device)

                valid_samples += 1

            if valid_samples == 0:
                return None, None


            min_valid_tokens = new_attention_mask.sum(dim=-1).min().item()
            if min_valid_tokens == 0:
                return None, None


            new_batch_dict = {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                "position_ids": new_position_ids,
                "responses": new_responses,
                "prompts": original_prompts.clone(),
            }

            new_tensor_dict = TensorDict(new_batch_dict, batch_size=[batch_size])

            meta_info = batch.meta_info.copy()

            fwd_batch = DataProto(
                batch=new_tensor_dict,
                non_tensor_batch={},
                meta_info=meta_info,
            )

            return fwd_batch, method_gt_mask

        except Exception:
            return None, None

    def _prepare_base_fwd_batch(self, batch: DataProto) -> Tuple[Optional[DataProto], Optional[torch.Tensor]]:
        try:
            batch_size = len(batch)


            original_input_ids = batch.batch["input_ids"]
            original_attention_mask = batch.batch["attention_mask"]
            original_position_ids = batch.batch["position_ids"]
            original_responses = batch.batch["responses"]
            original_prompts = batch.batch["prompts"]

            prompt_length = original_prompts.shape[-1]
            response_length = original_responses.shape[-1]
            device = original_input_ids.device


            new_input_ids = original_input_ids.clone()
            new_attention_mask = original_attention_mask.clone()
            new_position_ids = original_position_ids.clone()
            new_responses = torch.full_like(original_responses, self.tokenizer.pad_token_id or 0)
            method_gt_mask = torch.zeros((batch_size, response_length), dtype=torch.long, device=device)

            valid_samples = 0

            for i in range(batch_size):
                ground_truth = self._get_ground_truth(batch, i)
                method_gt = ground_truth.get("method_gt", "")


                if not method_gt:

                    valid_samples += 1
                    continue


                try:
                    method_gt_ids = self.tokenizer.encode(method_gt, add_special_tokens=False, return_tensors="pt")[0]
                except Exception:
                    valid_samples += 1
                    continue

                method_gt_len = len(method_gt_ids)

                if method_gt_len == 0:
                    valid_samples += 1
                    continue


                if method_gt_len > response_length:
                    method_gt_ids = method_gt_ids[:response_length]
                    method_gt_len = response_length


                pad_len = response_length - method_gt_len
                if pad_len > 0:
                    padded_response = torch.cat([
                        method_gt_ids,
                        torch.full((pad_len,), self.tokenizer.pad_token_id or 0, dtype=torch.long),
                    ])
                    response_mask = torch.cat([
                        torch.ones(method_gt_len, dtype=torch.long),
                        torch.zeros(pad_len, dtype=torch.long),
                    ])

                    method_mask = response_mask.clone()
                else:
                    padded_response = method_gt_ids
                    response_mask = torch.ones(response_length, dtype=torch.long)
                    method_mask = torch.ones(response_length, dtype=torch.long)


                new_responses[i] = padded_response.to(device)


                new_input_ids[i, prompt_length:] = padded_response.to(device)


                new_attention_mask[i, prompt_length:] = response_mask.to(device)


                full_mask = new_attention_mask[i]
                new_pos = (full_mask.cumsum(dim=-1) - 1).clamp(min=0)
                new_position_ids[i] = new_pos


                method_gt_mask[i] = method_mask.to(device)

                valid_samples += 1

            if valid_samples == 0:
                return None, None


            min_valid_tokens = new_attention_mask.sum(dim=-1).min().item()
            if min_valid_tokens == 0:
                return None, None


            new_batch_dict = {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                "position_ids": new_position_ids,
                "responses": new_responses,
                "prompts": original_prompts.clone(),
            }

            new_tensor_dict = TensorDict(new_batch_dict, batch_size=[batch_size])

            meta_info = batch.meta_info.copy()

            base_fwd_batch = DataProto(
                batch=new_tensor_dict,
                non_tensor_batch={},
                meta_info=meta_info,
            )

            return base_fwd_batch, method_gt_mask

        except Exception:
            return None, None

    def _compute_sequence_log_prob(
        self,
        token_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:

        masked_log_probs = token_log_probs * response_mask.float()
        seq_log_probs = masked_log_probs.sum(dim=-1)

        if normalize:
            seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
            seq_log_probs = seq_log_probs / seq_lengths

        return seq_log_probs

    def _compute_entropy_rewards(
        self,
        batch: DataProto,
        fwd_log_probs: Optional[torch.Tensor] = None,
        fwd_target_mask: Optional[torch.Tensor] = None,
        base_fwd_log_probs: Optional[torch.Tensor] = None,
        base_fwd_target_mask: Optional[torch.Tensor] = None,
        base_fwd_entropys: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size = len(batch)
        response_mask = batch.batch.get("response_mask", compute_response_mask(batch))
        response_length = response_mask.shape[-1]


        r_fwd = torch.zeros(batch_size)
        r_sem = torch.zeros(batch_size)


        information_gain = torch.zeros(batch_size)


        valid_cot_mask = torch.ones(batch_size, dtype=torch.bool)


        if (fwd_log_probs is not None and fwd_target_mask is not None and
            base_fwd_log_probs is not None and base_fwd_target_mask is not None):


            fwd_mask_valid = fwd_target_mask.sum(dim=-1) > 0
            base_mask_valid = base_fwd_target_mask.sum(dim=-1) > 0
            both_valid = fwd_mask_valid & base_mask_valid
            valid_cot_mask = both_valid

            cot_log_p = self._compute_sequence_log_prob(fwd_log_probs, fwd_target_mask, normalize=True)


            base_log_p = self._compute_sequence_log_prob(base_fwd_log_probs, base_fwd_target_mask, normalize=True)


            information_gain[both_valid] = cot_log_p[both_valid] - base_log_p[both_valid]


            r_fwd = information_gain.clone()

        elif fwd_log_probs is not None and fwd_target_mask is not None:

            fwd_mask_valid = fwd_target_mask.sum(dim=-1) > 0
            valid_cot_mask = fwd_mask_valid

            cot_log_p = self._compute_sequence_log_prob(fwd_log_probs, fwd_target_mask, normalize=True)
            r_fwd = cot_log_p.clone()
            r_fwd[~fwd_mask_valid] = 0.0


        if self.use_semantic_similarity:

            self._load_embedding_model()

            if self.use_semantic_similarity:

                pred_methods = []
                gt_methods = []
                prompt_texts = []

                prompts = batch.batch["prompts"]
                responses = batch.batch["responses"]
                attention_mask = batch.batch["attention_mask"]
                prompt_length = prompts.shape[-1]

                for i in range(batch_size):

                    valid_response_mask = attention_mask[i, prompt_length:]
                    valid_response_length = int(valid_response_mask.sum().item())
                    valid_response_ids = responses[i, :valid_response_length]
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)


                    _, method_pred = parse_model_output(response_str)
                    pred_methods.append(method_pred if method_pred else "")

                    prompt_ids = prompts[i]
                    prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                    prompt_texts.append(self._get_semantic_baseline_text(batch, i, prompt_str))
                    ground_truth = self._get_ground_truth(batch, i)
                    method_gt = ground_truth.get("method_gt", "")
                    gt_methods.append(method_gt if method_gt else "")


                if self.use_overview_only:

                    gen_overviews = []
                    gt_overviews = []

                    for i, method in enumerate(pred_methods):

                        overview, _ = self._extract_method_parts(method)
                        gen_overviews.append(overview)

                    for i, method in enumerate(gt_methods):
                        overview, _ = self._extract_method_parts(method)
                        gt_overviews.append(overview)


                    r_sem, _ = self._compute_contrastive_semantic_gain(
                        gen_overviews=gen_overviews,
                        gt_overviews=gt_overviews,
                        prompts=prompt_texts,
                    )
                else:

                    r_sem = self._compute_semantic_similarity(pred_methods, gt_methods)


        response_lengths = response_mask.sum(dim=-1)


        format_mask = torch.ones(batch_size)

        if self.use_format_penalty:
            prompts = batch.batch["prompts"]
            responses = batch.batch["responses"]
            attention_mask = batch.batch["attention_mask"]
            prompt_length = prompts.shape[-1]


            min_cot_length_hard = 1000
            min_cot_length_soft = 2500


            cot_lengths = []

            for i in range(batch_size):

                valid_response_mask = attention_mask[i, prompt_length:]
                valid_response_length = int(valid_response_mask.sum().item())
                valid_response_ids = responses[i, :valid_response_length]
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)


                cot_text, _ = parse_model_output(response_str)


                if not cot_text or len(cot_text.strip()) == 0:
                    format_mask[i] = 0.0

                    valid_cot_mask[i] = False
                    cot_lengths.append(0)
                    continue

                cot_length = len(cot_text.strip())
                cot_lengths.append(cot_length)


                if cot_length < min_cot_length_hard:
                    format_mask[i] = 0.0
                    valid_cot_mask[i] = False
                    continue


                if cot_length < min_cot_length_soft:

                    length_ratio = (cot_length - min_cot_length_hard) / (min_cot_length_soft - min_cot_length_hard)
                    length_penalty = 0.5 + 0.5 * length_ratio
                    format_mask[i] = length_penalty


                if self._check_cot_format_violation(cot_text):
                    format_mask[i] = 0.0


            if cot_lengths:
                cot_lengths_np = np.array(cot_lengths)

                if self.use_length_anchor and not self._length_anchor_initialized:

                    valid_lengths = cot_lengths_np[cot_lengths_np > 0]
                    if len(valid_lengths) > 0:
                        if self.target_cot_length is None:
                            self.target_cot_length = float(valid_lengths.mean())
                        self._length_anchor_initialized = True


                if self.use_length_anchor and self._length_anchor_initialized and self.target_cot_length is not None:
                    for i, cot_len in enumerate(cot_lengths):
                        if cot_len <= 0 or format_mask[i].item() == 0:
                            continue
                        elif cot_len < self.target_cot_length:

                            shortfall_ratio = (self.target_cot_length - cot_len) / self.target_cot_length
                            anchor_penalty = 1.0 - self.length_anchor_lambda * shortfall_ratio
                            anchor_penalty = max(0.0, min(1.0, anchor_penalty))

                            format_mask[i] = format_mask[i] * anchor_penalty


        r_eaig = torch.zeros(batch_size)

        if self.use_eaig:


            if base_fwd_log_probs is not None and fwd_log_probs is not None and base_fwd_target_mask is not None:

                valid_mask = base_fwd_target_mask.float()

                r_eaig, _ = self._compute_eaig_reward(
                    base_log_p=base_fwd_log_probs,
                    cot_log_p=fwd_log_probs,
                    valid_mask=valid_mask,
                    base_entropy=base_fwd_entropys,
                )


                if valid_cot_mask is not None:
                    n_invalid_eaig = (~valid_cot_mask).sum().item()
                    if n_invalid_eaig > 0:
                        r_eaig = r_eaig * valid_cot_mask.float()


        r_sem_shaped = r_sem.clone()


        if valid_cot_mask is not None:
            n_invalid_sem = (~valid_cot_mask).sum().item()
            if n_invalid_sem > 0:
                r_sem_shaped = r_sem_shaped * valid_cot_mask.float()

        if self.use_semantic_shaping and self.use_semantic_similarity:
            r_sem_shaped = self._shape_semantic_reward(r_sem_shaped)


        r_eaig_shaped = r_eaig.clone()
        if self.use_eaig_shaping and self.use_eaig:
            r_eaig_shaped = self._shape_eaig_reward(r_eaig)

        if valid_cot_mask is not None:
            n_invalid_eaig = (~valid_cot_mask).sum().item()
            if n_invalid_eaig > 0:
                r_eaig_shaped = r_eaig_shaped * valid_cot_mask.float()

        if self.use_eaig:
            total_reward = self.eaig_weight * r_eaig_shaped + self.semantic_weight * r_sem_shaped
        else:

            total_reward = self.semantic_weight * r_sem_shaped


        if self.use_format_penalty:
            total_reward = total_reward * format_mask


        reward_tensor = torch.zeros((batch_size, response_length), dtype=torch.float32)
        for i in range(batch_size):
            valid_len = int(response_mask[i].sum().item())
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] = total_reward[i]

        reward_extra_info = {
            "r_fwd": r_fwd.tolist(),
            "r_sem": r_sem.tolist(),
            "r_eaig": r_eaig.tolist(),
            "r_sem_shaped": r_sem_shaped.tolist(),
            "information_gain": information_gain.tolist(),
            "response_lengths": response_lengths.tolist(),
            "total_entropy_reward": total_reward.tolist(),
        }

        return reward_tensor, reward_extra_info

    def _compute_entropy_rewards_for_eval(self, batch: DataProto):
        fwd_log_probs = None
        fwd_target_mask = None
        base_fwd_log_probs = None
        base_fwd_target_mask = None
        base_fwd_entropys = None

        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )

        if self.use_entropy_reward:
            try:
                fwd_batch, method_gt_mask = self._prepare_fwd_batch(batch)
                if fwd_batch is not None and method_gt_mask is not None:
                    fwd_batch_padded, pad_size = pad_dataproto_to_divisor(fwd_batch, size_divisor)
                    fwd_log_prob_output = self.actor_rollout_wg.compute_log_prob(fwd_batch_padded)
                    fwd_log_prob_output = unpad_dataproto(fwd_log_prob_output, pad_size=pad_size)
                    fwd_log_probs = fwd_log_prob_output.batch["old_log_probs"]
                    fwd_target_mask = method_gt_mask
            except Exception:
                pass

            try:
                base_fwd_batch, base_method_gt_mask = self._prepare_base_fwd_batch(batch)
                if base_fwd_batch is not None and base_method_gt_mask is not None:
                    base_fwd_batch.meta_info["calculate_entropy"] = self.use_eaig
                    base_fwd_batch_padded, pad_size = pad_dataproto_to_divisor(base_fwd_batch, size_divisor)
                    if self.use_reference_policy:
                        ref_wg = self.actor_rollout_wg if self.ref_in_actor else self.ref_policy_wg
                        base_fwd_log_prob_output = ref_wg.compute_ref_log_prob(base_fwd_batch_padded)
                        base_fwd_log_prob_output = unpad_dataproto(base_fwd_log_prob_output, pad_size=pad_size)
                        base_fwd_log_probs = base_fwd_log_prob_output.batch.get(
                            "ref_log_prob",
                            base_fwd_log_prob_output.batch.get("old_log_probs"),
                        )
                        base_fwd_entropys = base_fwd_log_prob_output.batch.get("ref_entropy", None)
                    else:
                        base_fwd_log_prob_output = self.actor_rollout_wg.compute_log_prob(base_fwd_batch_padded)
                        base_fwd_log_prob_output = unpad_dataproto(base_fwd_log_prob_output, pad_size=pad_size)
                        base_fwd_log_probs = base_fwd_log_prob_output.batch["old_log_probs"]
                    base_fwd_target_mask = base_method_gt_mask
            except Exception:
                pass

        return self._compute_entropy_rewards(
            batch=batch,
            fwd_log_probs=fwd_log_probs,
            fwd_target_mask=fwd_target_mask,
            base_fwd_log_probs=base_fwd_log_probs,
            base_fwd_target_mask=base_fwd_target_mask,
            base_fwd_entropys=base_fwd_entropys,
        )

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")

            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            if "response_mask" not in test_batch.batch.keys():
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)

            if self.use_entropy_reward:
                reward_tensor, reward_extra_infos = self._compute_entropy_rewards_for_eval(test_batch)
            else:
                reward_tensor, reward_extra_infos = compute_reward(test_batch, self.val_reward_fn)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            if reward_extra_infos:
                for key, lst in reward_extra_infos.items():
                    reward_extra_infos_dict[key].extend(lst)

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()


        if (self.use_entropy_reward or self.val_reward_fn is not None) and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)


                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):

                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))


                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)


                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()


                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob_output.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob_output.batch.pop("entropys")
                        batch = batch.union(old_log_prob_output)


                    fwd_log_probs = None
                    fwd_target_mask = None
                    base_fwd_log_probs = None
                    base_fwd_target_mask = None

                    if self.use_entropy_reward:


                        with marked_timer("fwd_log_prob", timing_raw, color="magenta"):
                            try:
                                fwd_batch, method_gt_mask = self._prepare_fwd_batch(batch)

                                if fwd_batch is not None and method_gt_mask is not None:

                                    min_valid_tokens = fwd_batch.batch["attention_mask"].sum(dim=-1).min().item()
                                    if min_valid_tokens > 0:

                                        fwd_log_prob_output = self.actor_rollout_wg.compute_log_prob(fwd_batch)
                                        fwd_log_probs = fwd_log_prob_output.batch["old_log_probs"]
                                        fwd_target_mask = method_gt_mask
                            except Exception:
                                pass


                            with marked_timer("base_fwd_log_prob", timing_raw, color="green"):
                                try:
                                    base_fwd_batch, base_method_gt_mask = self._prepare_base_fwd_batch(batch)


                                    calculate_entropy_for_eaig = self.use_eaig
                                    base_fwd_entropys = None

                                    if base_fwd_batch is not None and base_method_gt_mask is not None:

                                        min_valid_tokens = base_fwd_batch.batch["attention_mask"].sum(dim=-1).min().item()
                                        if min_valid_tokens > 0:
                                            if self.use_reference_policy:
                                                try:
                                                    base_fwd_batch.meta_info["calculate_entropy"] = calculate_entropy_for_eaig
                                                    ref_wg = self.actor_rollout_wg if self.ref_in_actor else self.ref_policy_wg
                                                    base_fwd_log_prob_output = ref_wg.compute_ref_log_prob(base_fwd_batch)
                                                    base_fwd_log_probs = base_fwd_log_prob_output.batch.get(
                                                        "ref_log_prob",
                                                        base_fwd_log_prob_output.batch.get("old_log_probs")
                                                    )
                                                    base_fwd_target_mask = base_method_gt_mask


                                                    if calculate_entropy_for_eaig:
                                                        base_fwd_entropys = base_fwd_log_prob_output.batch.get("ref_entropy", None)

                                                except Exception:
                                                    base_fwd_log_prob_output = self.actor_rollout_wg.compute_log_prob(base_fwd_batch)
                                                    base_fwd_log_probs = base_fwd_log_prob_output.batch["old_log_probs"]
                                                    base_fwd_target_mask = base_method_gt_mask
                                            else:

                                                base_fwd_log_prob_output = self.actor_rollout_wg.compute_log_prob(base_fwd_batch)
                                                base_fwd_log_probs = base_fwd_log_prob_output.batch["old_log_probs"]
                                                base_fwd_target_mask = base_method_gt_mask
                                except Exception:
                                    pass


                    with marked_timer("reward", timing_raw, color="yellow"):

                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.use_entropy_reward:

                            reward_tensor, reward_extra_infos_dict = self._compute_entropy_rewards(
                                batch=batch,
                                fwd_log_probs=fwd_log_probs,
                                fwd_target_mask=fwd_target_mask,
                                base_fwd_log_probs=base_fwd_log_probs,
                                base_fwd_target_mask=base_fwd_target_mask,
                                base_fwd_entropys=base_fwd_entropys,
                            )
                            batch.batch["token_level_scores"] = reward_tensor
                        else:

                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})


                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)


                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)


                    with marked_timer("adv", timing_raw, color="brown"):

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]


                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )


                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)


                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)


                        self.global_step += 1


                    if (
                        (self.use_entropy_reward or self.val_reward_fn is not None)
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)


                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()


                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    progress_bar.close()
                    return
