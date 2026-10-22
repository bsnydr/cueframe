"""Generated stage → sync → verify → plan → cut → pixel oracle, plus the offset-only control."""

import json

import pytest

from cueframe.clock import AffineClock
from cueframe.cut import cut
from cueframe.media import decode_audio, probe
from cueframe.oracle import flash_oracle
from cueframe.plan import Camera, make_plan
from cueframe.sync import SyncConfig, sync_camera
from cueframe.verify import verify_camera

pytestmark = [pytest.mark.ffmpeg, pytest.mark.slow]


@pytest.fixture(scope="module")
def synced(quick_stage):
    inputs = json.loads((quick_stage / "inputs.json").read_text())
    cfg = SyncConfig()
    ref = decode_audio(quick_stage / inputs["reference"], rate=cfg.analysis_rate)
    cams = {c["name"]: decode_audio(quick_stage / c["path"], rate=cfg.analysis_rate)
            for c in inputs["cameras"]}
    syncs = {n: sync_camera(cams[n], ref, cfg.analysis_rate, name=n, cfg=cfg) for n in cams}
    return inputs, ref, cams, syncs


def truth(quick_stage):
    manifest = json.loads((quick_stage / "manifest.json").read_text())
    clocks = {c["name"]: AffineClock.from_json(c["true_clock"]) for c in manifest["cameras"]}
    return manifest, clocks


def test_generated_media_invariants(quick_stage):
    manifest, _ = truth(quick_stage)
    for c in manifest["cameras"]:
        p = probe(quick_stage / c["path"])
        assert str(p.fps) == c["fps"]
        assert p.audio_rate == manifest["sample_rate"]
        assert (p.width, p.height) == tuple(manifest["size"])
        assert abs(p.nb_frames / p.fps - c["duration_s"]) <= 1 / p.fps
    assert len(manifest["flash_events_ref_s"]) == 8
    import numpy as np

    gaps = np.diff(manifest["flash_events_ref_s"])
    assert gaps.min() > 1.0 and gaps.std() > 0.05  # irregular, not periodic


def test_estimates_match_planted_clocks(quick_stage, synced):
    _, clocks = truth(quick_stage)
    _, _, _, syncs = synced
    for name, s in syncs.items():
        assert s.status == "ok", (name, s.reason)
        t = clocks[name]
        assert abs(s.clock.offset - t.offset) * 1000 < 1.5, name
        assert abs(s.clock.drift_ppm - t.drift_ppm) < 60.0, name


def test_heldout_verification_passes(synced):
    _, ref, cams, syncs = synced
    for name, s in syncs.items():
        v = verify_camera(cams[name], ref, SyncConfig().analysis_rate, s.clock, name=name)
        assert v.status == "pass", v.to_json()


def cameras_for(quick_stage, inputs, clocks):
    cams = []
    for c in inputs["cameras"]:
        p = probe(quick_stage / c["path"])
        cams.append(Camera(c["name"], str(quick_stage / c["path"]), p.fps,
                           float(p.nb_frames / p.fps), p.width, p.height, clocks[c["name"]]))
    return cams


def test_cut_matches_oracle_and_offset_only_control_fails(quick_stage, synced, tmp_path):
    inputs, _, _, syncs = synced
    manifest, _ = truth(quick_stage)
    estimated = {n: s.clock for n, s in syncs.items()}
    plan = make_plan(cameras_for(quick_stage, inputs, estimated), fps=25, size=(320, 180), seed=1)
    plan.save(tmp_path / "plan.json")
    result = cut(plan, quick_stage / inputs["reference"], tmp_path / "edit.mp4")
    p = probe(result.out_path)
    assert p.nb_frames == plan.n_frames
    assert abs(p.duration - plan.duration) <= 0.04 + 0.05
    assert p.audio_rate == manifest["sample_rate"]
    report = flash_oracle(quick_stage / "manifest.json", tmp_path / "plan.json",
                          result.out_path)
    assert report["status"] == "pass", report
    assert report["n_events"] >= 4

    # negative control: a single clap sync at the first fit window, drift ignored → late
    # flashes land on the wrong frame
    offset_only = {n: s.clock.clap_sync(min(w.t_cam for w in s.windows if w.used))
                   for n, s in syncs.items()}
    plan2 = make_plan(cameras_for(quick_stage, inputs, offset_only), fps=25, size=(320, 180),
                      seed=1, start=plan.timeline_start, duration=plan.duration)
    plan2.save(tmp_path / "plan2.json")
    cut(plan2, quick_stage / inputs["reference"], tmp_path / "edit2.mp4")
    report2 = flash_oracle(quick_stage / "manifest.json", tmp_path / "plan2.json",
                           tmp_path / "edit2.mp4")
    assert report2["status"] == "fail", report2
    failed = [e for e in report2["events"] if not e["ok"]]
    assert failed and max(e["t_ref"] for e in failed) > manifest["flash_events_ref_s"][2]
