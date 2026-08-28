from chronotrace.data import build_probes, build_stage_examples, build_worlds


def test_world_generation_is_deterministic() -> None:
    assert build_worlds(7, 12) == build_worlds(7, 12)
    assert build_worlds(7, 12) != build_worlds(8, 12)


def test_stage_examples_cover_same_worlds_without_stage_labels_in_prompt() -> None:
    worlds = build_worlds(9, 12)
    stage_a = build_stage_examples(worlds, "A")
    stage_b = build_stage_examples(worlds, "B")

    assert {row.world_id for row in stage_a} == {row.world_id for row in stage_b}
    assert len(stage_a) == len(stage_b) == 36
    assert all("stage A" not in row.prompt and "stage B" not in row.prompt for row in stage_a)
    assert all("stage A" not in row.prompt and "stage B" not in row.prompt for row in stage_b)


def test_binding_pairs_hold_query_target_fixed_and_change_only_context() -> None:
    worlds = build_worlds(11, 12)
    probes = build_probes(worlds, n_decoys=3)
    by_id = {probe.probe_id: probe for probe in probes}

    for world in worlds:
        congruent = by_id[f"{world.world_id}-a2b-congruent"]
        incongruent = by_id[f"{world.world_id}-a2b-incongruent"]
        assert congruent.answer == incongruent.answer
        assert congruent.decoys == incongruent.decoys
        assert world.entity in congruent.prompt
        assert world.entity in incongruent.prompt
        assert congruent.prompt != incongruent.prompt

        reverse_congruent = by_id[f"{world.world_id}-b2a-congruent"]
        reverse_incongruent = by_id[f"{world.world_id}-b2a-incongruent"]
        assert reverse_congruent.answer == reverse_incongruent.answer
        assert reverse_congruent.decoys == reverse_incongruent.decoys
        assert world.alias in reverse_congruent.prompt
        assert world.alias in reverse_incongruent.prompt
        assert reverse_congruent.prompt != reverse_incongruent.prompt


def test_probe_answer_is_not_present_in_control_prompt() -> None:
    worlds = build_worlds(13, 12)
    probes = build_probes(worlds, n_decoys=3)
    for probe in probes:
        if probe.family in {"a_control", "b_control"}:
            assert probe.answer.strip(" .") not in probe.prompt
