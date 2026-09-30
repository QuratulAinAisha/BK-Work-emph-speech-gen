"""Audit update boundaries without loading a full model. / 전체 모델 없이 갱신 범위를 검사합니다."""

import copy
import unittest

import torch

from scripts.audit_planner_checkpoint import audit_checkpoints, UNIT_ARCHITECTURE, CONTINUOUS_ARCHITECTURE


def checkpoint(architecture=UNIT_ARCHITECTURE):
    state = {'semantic_planner.output.weight': torch.ones(3, 2),
             'length_predictor.weight': torch.ones(2), 'codec_generator.weight': torch.ones(2)}
    if architecture == UNIT_ARCHITECTURE:
        state['semantic_planner.codebook.centers'] = torch.ones(3, 768)
    return {'architecture': architecture, 'state_dict': state,
            'optimizer': {'state': {0: {'exp_avg': torch.zeros(2)}}}}


class PlannerCheckpointAuditTests(unittest.TestCase):
    def test_baseline_allows_planner_and_length_but_prior_rejects_length(self):
        initial = checkpoint()
        changed = copy.deepcopy(initial)
        changed['state_dict']['semantic_planner.output.weight'].add_(1)
        changed['state_dict']['length_predictor.weight'].add_(1)
        self.assertTrue(audit_checkpoints(initial, changed)['passed'])
        self.assertFalse(audit_checkpoints(initial, changed, unit_prior=True)['passed'])

    def test_frozen_changes_and_unit_codebook_changes_rejected(self):
        for key in ('codec_generator.weight', 'semantic_planner.codebook.centers'):
            initial = checkpoint()
            changed = copy.deepcopy(initial)
            changed['state_dict'][key].add_(1)
            self.assertFalse(audit_checkpoints(initial, changed)['passed'])

    def test_memory_experiment_freezes_duration_without_claiming_prior_training(self):
        initial = checkpoint()
        changed = copy.deepcopy(initial)
        changed['state_dict']['semantic_planner.output.weight'].add_(1)
        result = audit_checkpoints(initial, changed, planner_only=True)
        self.assertTrue(result['passed'])
        self.assertFalse(result['unit_prior'])
        changed['state_dict']['length_predictor.weight'].add_(1)
        self.assertFalse(audit_checkpoints(initial, changed, planner_only=True)['passed'])

    def test_continuous_replacement_allows_only_planner_key_and_shape_changes(self):
        initial, candidate = checkpoint(CONTINUOUS_ARCHITECTURE), checkpoint()
        initial['state_dict']['semantic_planner.output.weight'] = torch.ones(768, 2)
        initial['state_dict']['semantic_planner.old.weight'] = torch.ones(2)
        candidate['state_dict']['semantic_planner.mask_embedding'] = torch.zeros(768)
        self.assertTrue(audit_checkpoints(initial, candidate)['passed'])
        del candidate['state_dict']['codec_generator.weight']
        self.assertFalse(audit_checkpoints(initial, candidate)['passed'])

    def test_nonfinite_optimizer_rejected_and_unit_key_changes_rejected(self):
        initial = checkpoint()
        changed = copy.deepcopy(initial)
        changed['optimizer']['state'][0]['exp_avg'][0] = float('nan')
        result = audit_checkpoints(initial, changed)
        self.assertFalse(result['all_tensors_finite'])
        self.assertIn('checkpoint.optimizer.state.0.exp_avg', result['nonfinite_tensor_paths'])
        changed = copy.deepcopy(initial)
        changed['state_dict']['semantic_planner.new'] = torch.ones(1)
        self.assertFalse(audit_checkpoints(initial, changed)['passed'])


if __name__ == '__main__':
    unittest.main()
