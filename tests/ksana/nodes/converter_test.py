import unittest

import torch

from ksana.nodes.convert import convert_text_embeds_to_ksana


class TestConvertTextEmbedsToKsana(unittest.TestCase):
    """test convert_text_embeds_to_ksana function"""

    def test(self):
        emb_len = 4096
        positive_embeds = [
            torch.randn(10, emb_len),  # shape: (seq_len, emb_len)
            torch.randn(20, emb_len),
        ]
        negative_embeds = [
            torch.randn(15, emb_len),
        ]

        text_embeds = {
            "prompt_embeds": positive_embeds,
            "negative_prompt_embeds": negative_embeds,
        }

        positive_result, negative_result = convert_text_embeds_to_ksana(text_embeds)
        positive_result = positive_result[0][0]
        negative_result = negative_result[0][0]
        # check shape
        self.assertEqual(positive_result.shape, (len(positive_embeds), 20, emb_len))
        self.assertEqual(negative_result.shape, (len(negative_embeds), 20, emb_len))
        # check padding
        self.assertTrue(torch.all(positive_result[0, 10:, :] == 0))
        self.assertTrue(torch.all(negative_result[0, 15:, :] == 0))


if __name__ == "__main__":
    unittest.main()
