import copy
import importlib.util
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TORCH_FUNCTIONAL_PATH = REPO_ROOT / "verl" / "utils" / "torch_functional.py"


@pytest.fixture(scope="module")
def scheduler_module():
    spec = importlib.util.spec_from_file_location(
        "_torch_functional_scheduler_contract",
        TORCH_FUNCTIONAL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_constant_warmup_update_changes_parameter(scheduler_module):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=8e-3, weight_decay=0.0)
    scheduler = scheduler_module.get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=8,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    before = parameter.detach().clone()
    parameter.square().sum().backward()
    optimizer.step()

    assert not torch.equal(parameter.detach(), before)
    assert (before - parameter.detach()).item() == pytest.approx(1e-3, rel=1e-4)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-3)


@pytest.mark.parametrize("schedule_name", ["cosine", "wsd"])
def test_other_warmup_schedules_start_at_first_fraction(scheduler_module, schedule_name):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.8)
    if schedule_name == "cosine":
        scheduler_module.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=8,
            num_training_steps=20,
        )
    else:
        scheduler_module.get_wsd_schedule_with_warmup(
            optimizer,
            num_warmup_steps=8,
            num_training_steps=20,
        )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_scheduler_state_dict_resume_does_not_restart_warmup(scheduler_module):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.8)
    scheduler = scheduler_module.get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=8,
    )

    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    saved_parameter = parameter.detach().clone()
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    assert scheduler_state["last_epoch"] == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.4)

    resumed_parameter = torch.nn.Parameter(saved_parameter.clone())
    resumed_optimizer = torch.optim.SGD([resumed_parameter], lr=0.8)
    resumed_scheduler = scheduler_module.get_constant_schedule_with_warmup(
        resumed_optimizer,
        num_warmup_steps=8,
    )
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)

    assert resumed_scheduler.last_epoch == 3
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.4)
    before_resume_update = resumed_parameter.detach().clone()
    resumed_parameter.grad = torch.ones_like(resumed_parameter)
    resumed_optimizer.step()
    resumed_optimizer.zero_grad()
    resumed_scheduler.step()

    assert (before_resume_update - resumed_parameter.detach()).item() == pytest.approx(0.4)
    assert resumed_scheduler.last_epoch == 4
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
