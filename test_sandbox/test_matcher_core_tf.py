import numpy as np
import tensorflow.compat.v1 as tf
import os,sys,json
from pathlib import Path

PATH =Path(os.getcwd()).parent
sys.path.append(str(PATH.absolute()) + "/")
PATH =Path(os.getcwd())
sys.path.append(str(PATH.absolute()) + "/")
from mmdet.core_tf.bbox.matcher import max_iou_matcher as argmax_matcher
from mmdet.core_tf.builder import build_matcher
from mmdet.utils import test_case 


class ArgMaxMatcherTest(test_case.TestCase):



                    # pos_iou_thr,
                    # neg_iou_thr,
                    # min_pos_iou=.0,
                    # gt_max_assign_all=True,
                    # ignore_iof_thr=-1,
                    # ignore_wrt_candidates=True,
                    # match_low_quality=True,
                    # iou_calculator=dict(type='IouSimilarity')
  def test_build_cfg(self):

    def graph_fn(similarity):
      cfg = dict(type='ArgMaxMatcher',pos_iou_thr=3.,neg_iou_thr=3.)
      matcher = build_matcher(cfg)
      match = matcher.match(similarity)
      matched_cols = match.matched_column_indicator()
      unmatched_cols = match.unmatched_column_indicator()
      match_results = match.match_results
      return (matched_cols, unmatched_cols, match_results)

    similarity = np.array([[1, 1, 1, 3, 1],
                           [2, -1, 2, 0, 4],
                           [3, 0, -1, 0, 0]], dtype=np.float32)
    expected_matched_cols = np.array([0, 3, 4])
    expected_matched_rows = np.array([2, 0, 1])
    expected_unmatched_cols = np.array([1, 2])

    (res_matched_cols, res_unmatched_cols,
     match_results) = self.execute(graph_fn, [similarity])
    self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
    self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
    self.assertAllEqual(np.nonzero(res_unmatched_cols)[0],
                        expected_unmatched_cols)
  def test_return_correct_matches_with_matched_threshold(self):

    def graph_fn(similarity):
      matcher = argmax_matcher.ArgMaxMatcher(pos_iou_thr=3.,neg_iou_thr=3.)
      match = matcher.match(similarity)
      matched_cols = match.matched_column_indicator()
      unmatched_cols = match.unmatched_column_indicator()
      match_results = match.match_results
      return (matched_cols, unmatched_cols, match_results)

    similarity = np.array([[1, 1, 1, 3, 1],
                           [2, -1, 2, 0, 4],
                           [3, 0, -1, 0, 0]], dtype=np.float32)
    expected_matched_cols = np.array([0, 3, 4])
    expected_matched_rows = np.array([2, 0, 1])
    expected_unmatched_cols = np.array([1, 2])

    (res_matched_cols, res_unmatched_cols,
     match_results) = self.execute(graph_fn, [similarity])
    self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
    self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
    self.assertAllEqual(np.nonzero(res_unmatched_cols)[0],
                        expected_unmatched_cols)

  def test_return_correct_matches_with_matched_and_unmatched_threshold(self):

    def graph_fn(similarity):
      matcher = argmax_matcher.ArgMaxMatcher(pos_iou_thr=3.,neg_iou_thr=2.)
      match = matcher.match(similarity)
      matched_cols = match.matched_column_indicator()
      unmatched_cols = match.unmatched_column_indicator()
      match_results = match.match_results
      return (matched_cols, unmatched_cols, match_results)

    similarity = np.array([[1, 1, 1, 3, 1],
                           [2, -1, 2, 0, 4],
                           [3, 0, -1, 0, 0]], dtype=np.float32)
    expected_matched_cols = np.array([0, 3, 4])
    expected_matched_rows = np.array([2, 0, 1])
    expected_unmatched_cols = np.array([1])  # col 2 has too high maximum val

    (res_matched_cols, res_unmatched_cols,
     match_results) = self.execute(graph_fn, [similarity])
    self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
    self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
    self.assertAllEqual(np.nonzero(res_unmatched_cols)[0],
                        expected_unmatched_cols)



  def test_return_correct_matches_unmatched_row_not_using_force_match(self):

    def graph_fn(similarity):
      matcher = argmax_matcher.ArgMaxMatcher(pos_iou_thr=3.,
                                             neg_iou_thr=2., match_low_quality=False)
      match = matcher.match(similarity)
      matched_cols = match.matched_column_indicator()
      unmatched_cols = match.unmatched_column_indicator()
      match_results = match.match_results
      return (matched_cols, unmatched_cols, match_results)

    similarity = np.array([[1, 1, 1, 3, 1],
                           [-1, 0, -2, -2, -1],
                           [3, 0, -1, 2, 0]], dtype=np.float32)
    expected_matched_cols = np.array([0, 3])
    expected_matched_rows = np.array([2, 0])
    expected_unmatched_cols = np.array([1, 2, 4])

    (res_matched_cols, res_unmatched_cols,
     match_results) = self.execute(graph_fn, [similarity])
    self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
    self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
    self.assertAllEqual(np.nonzero(res_unmatched_cols)[0],
                        expected_unmatched_cols)



  def test_return_correct_matches_unmatched_row_while_using_force_match(self):
    def graph_fn(similarity):
      matcher = argmax_matcher.ArgMaxMatcher(pos_iou_thr=3.,
                                             neg_iou_thr=2.,
                                             match_low_quality=True)
      match = matcher.match(similarity)
      matched_cols = match.matched_column_indicator()
      unmatched_cols = match.unmatched_column_indicator()
      match_results = match.match_results
      return (matched_cols, unmatched_cols, match_results)

    similarity = np.array([[1, 1, 1, 3, 1],
                           [-1, 0, -2, -2, -1],
                           [3, 0, -1, 2, 0]], dtype=np.float32)
    expected_matched_cols = np.array([0, 1, 3])
    expected_matched_rows = np.array([2, 1, 0])
    expected_unmatched_cols = np.array([2, 4])  # col 2 has too high max val

    (res_matched_cols, res_unmatched_cols,
     match_results) = self.execute(graph_fn, [similarity])
    self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
    self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
    self.assertAllEqual(np.nonzero(res_unmatched_cols)[0],
                        expected_unmatched_cols)

  def test_return_correct_matches_using_force_match_padded_groundtruth(self):
    def graph_fn(similarity, valid_rows):
      matcher = argmax_matcher.ArgMaxMatcher(pos_iou_thr=3.,
                                             neg_iou_thr=2.,
                                             match_low_quality=True)
      match = matcher.match(similarity, valid_rows)
      matched_cols = match.matched_column_indicator()
      unmatched_cols = match.unmatched_column_indicator()
      match_results = match.match_results
      return (matched_cols, unmatched_cols, match_results)

    similarity = np.array([[1, 1, 1, 3, 1],
                           [-1, 0, -2, -2, -1],
                           [0, 0, 0, 0, 0],
                           [3, 0, -1, 2, 0],
                           [0, 0, 0, 0, 0]], dtype=np.float32)
    valid_rows = np.array([True, True, False, True, False])
    expected_matched_cols = np.array([0, 1, 3])
    expected_matched_rows = np.array([3, 1, 0])
    expected_unmatched_cols = np.array([2, 4])  # col 2 has too high max val

    (res_matched_cols, res_unmatched_cols,
     match_results) = self.execute(graph_fn, [similarity, valid_rows])
    self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
    self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
    self.assertAllEqual(np.nonzero(res_unmatched_cols)[0],
                        expected_unmatched_cols)


if __name__ == '__main__':
  tf.test.main()