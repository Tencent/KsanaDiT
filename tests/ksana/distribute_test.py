import unittest

from ksana.utils.distribute import get_rank_id_result


class TestDistributeGetResult(unittest.TestCase):

    def test_dict(self):
        results = [{1: "a"}, {2: "b"}, {0: "c"}, {4: None}]
        func_res = get_rank_id_result(results)
        self.assertEqual(func_res, "c")

        results = [{1: "a"}, {2: "b"}, {0: [4, 5]}, {4: None}]
        func_res = get_rank_id_result(results)
        self.assertEqual(func_res, [4, 5])

        results = [{1: "a"}, {2: "b"}, {0: [4, 5]}, {4: None}]
        func_res = get_rank_id_result(results, rank_id=2)
        self.assertEqual(func_res, "b")

    def test_any_other(self):
        results = ["a", "b", "c"]
        func_res = get_rank_id_result(results)
        self.assertEqual(func_res, "a")

        func_res = get_rank_id_result(results, rank_id=2)
        self.assertEqual(func_res, "c")

        func_res = get_rank_id_result(results, rank_id=10)
        self.assertEqual(func_res, "c")

        results = [["x", "y"], ["z", "w"], ["a", "b"]]
        func_res = get_rank_id_result(results)
        self.assertEqual(func_res, ["x", "y"])

        results = [["x", "y"], ["z", "w"], ["a", "b"]]
        func_res = get_rank_id_result(results, rank_id=1)
        self.assertEqual(func_res, ["z", "w"])

        results = [["x", "y"], ["z", "w"], ["a", "b"]]
        func_res = get_rank_id_result(results, rank_id=10)
        self.assertEqual(func_res, ["a", "b"])

    def test_has_none(self):

        results = [{1: 2}, {2: 2}, {0: None}, {4: None}]
        self.assertEqual(get_rank_id_result(results), None)

        with self.assertRaises(ValueError):
            results = [{1: 2}, {2: 2}, {0: None}, {4: None}]
            get_rank_id_result(results, check_no_none_res=True)
        with self.assertRaises(ValueError):
            results = [{1: 2}, {2: 2}, {0: [5, None]}, {4: None}]
            get_rank_id_result(results, check_no_none_res=True)


if __name__ == "__main__":
    unittest.main()
