import os
import unittest

import psutil
from nodes_test_helper import COMFY_MODEL_DIFFUSION_ROOT, TEST_STEPS, iter_test_models, run_load_and_generate

from ksana import KsanaAttentionBackend
from ksana.accelerator import platform
from ksana.config import KsanaTorchCompileConfig
from ksana.utils import get_rank_id, log


class TestAttentionsForAllModels(unittest.TestCase):

    def run_once(self, model_name, image_latent_shape, text_shape, expected_model_key, attn_backend):
        load_output, generate_output = run_load_and_generate(
            os.path.join(COMFY_MODEL_DIFFUSION_ROOT, model_name),
            image_latent_shape,
            text_shape,
            TEST_STEPS,
            attn_backend=attn_backend,
            torch_compile_config=KsanaTorchCompileConfig(),
        )
        self.assertEqual(load_output.model, expected_model_key)
        generate_output = generate_output.samples
        if get_rank_id() == 0:
            # only return tensor on rank 0
            self.assertIsNotNone(generate_output)
        else:
            self.assertIsNone(generate_output)

    def _get_rss_memory_usage_in_gb(self):
        process = psutil.Process(os.getpid())
        # RSS (Resident Set Size) 包含所有常驻内存
        rss_gb = process.memory_info().rss / 1024 / 1024 / 1024
        return rss_gb

    def test_all_attention_backend(self):
        # TODO(rockcao): support SAGE_SLA test on TurboWan model
        exclude_list = [KsanaAttentionBackend.SAGE_SLA]
        init_rss_memory_gb = self._get_rss_memory_usage_in_gb()
        log.info(f"初始内存使用: {init_rss_memory_gb:.2f} GB")
        for model_name, img_shape, text_shape, expected_model_key in iter_test_models():
            for attn_backend in KsanaAttentionBackend.get_supported_list(exclude=exclude_list):
                print(f"-----------------test {model_name} {attn_backend} -----------------")
                self.run_once(model_name, img_shape, text_shape, expected_model_key, attn_backend)
                after_rss_memory_gb = self._get_rss_memory_usage_in_gb()
                log.info(f"测试 {model_name} {attn_backend} 后内存使用: {after_rss_memory_gb:.2f} GB")
                memory_diff_gb = after_rss_memory_gb - init_rss_memory_gb
                log.info(f"测试 {model_name} {attn_backend} 内存增量: {memory_diff_gb:.2f} GB")
                if platform.is_gpu():  # TODO(qian): 在npu的时候模型权重在当前test没有调用析构函数释放。
                    self.assertLessEqual(memory_diff_gb, 100.0, f"内存增量过大: {memory_diff_gb:.2f} GB")


if __name__ == "__main__":
    unittest.main()
