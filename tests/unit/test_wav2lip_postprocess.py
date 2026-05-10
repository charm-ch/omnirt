from __future__ import annotations

import numpy as np

from omnirt.models.wav2lip.postprocess import (
    BlendConfig,
    MouthGeometry,
    blend_mouth_patch_easy,
    blend_mouth_patch_basic,
    blend_mouth_patch,
    build_easy_mouth_blend_mask,
    build_mouth_blend_mask,
    build_jaw_motion_mask,
    metadata_face_box_to_crop,
    metadata_radius_to_input_crop,
    resize_reference_frame,
    select_wav2lip_model_crop,
)


def test_easy_mask_matches_lip_polygon_and_feathers_edges() -> None:
    geometry = MouthGeometry(
        center=(192, 220),
        rx=42,
        ry=14,
        outer_lip=((150, 220), (170, 206), (192, 202), (214, 206), (234, 220), (214, 234), (192, 238), (170, 234)),
    )

    mask = build_easy_mouth_blend_mask((384, 384), geometry, mask_dilation=2.5, mask_feathering=2.0)

    assert mask.shape == (384, 384, 1)
    assert float(mask[220, 192, 0]) > 0.85
    assert float(mask[0, 0, 0]) == 0.0
    assert 0.0 < float(mask[300, 192, 0]) < 0.85


def test_easy_blend_keeps_non_mouth_pixels_from_original() -> None:
    original = np.full((384, 384, 3), [120, 130, 140], dtype=np.uint8)
    pred = np.full((384, 384, 3), [20, 50, 220], dtype=np.uint8)
    geometry = MouthGeometry(
        center=(192, 220),
        rx=42,
        ry=14,
        outer_lip=((150, 220), (170, 206), (192, 202), (214, 206), (234, 220), (214, 234), (192, 238), (170, 234)),
    )

    blended = blend_mouth_patch_easy(pred, original, geometry=geometry, mask_dilation=2.5, mask_feathering=2.0)

    assert np.array_equal(blended[0, 0], original[0, 0])
    assert np.mean(np.abs(blended[220, 192].astype(np.int16) - pred[220, 192].astype(np.int16))) < 20


def test_easy_mask_does_not_bleed_into_distant_face_or_crop_edges() -> None:
    geometry = MouthGeometry(
        center=(192, 220),
        rx=42,
        ry=14,
        outer_lip=((150, 220), (170, 206), (192, 202), (214, 206), (234, 220), (214, 234), (192, 238), (170, 234)),
    )

    mask = build_easy_mouth_blend_mask((384, 384), geometry, mask_dilation=2.5, mask_feathering=2.0)

    assert float(mask[220, 192, 0]) > 0.85
    assert float(mask[80, 192, 0]) < 0.02
    assert float(mask[330, 192, 0]) < 0.02
    assert float(mask[:, 0, 0].max()) == 0.0
    assert float(mask[:, -1, 0].max()) == 0.0
    assert float(mask[0, :, 0].max()) == 0.0
    assert float(mask[-1, :, 0].max()) == 0.0


def test_enhanced_mask_expands_lower_lip_without_reaching_chin() -> None:
    geometry = MouthGeometry(
        center=(50, 50),
        rx=20,
        ry=8,
        outer_lip=((30, 49), (40, 44), (50, 43), (60, 44), (70, 49), (62, 55), (50, 58), (38, 55)),
        inner_mouth=((38, 50), (50, 49), (62, 50), (50, 53)),
    )

    tight = build_mouth_blend_mask(
        (100, 100),
        geometry,
        BlendConfig(lower_lip_expand=0.0, lower_lip_dynamic_expand=0.0, corner_expand=0.0),
    )
    expanded = build_mouth_blend_mask((100, 100), geometry, BlendConfig(lower_lip_expand=0.55, corner_expand=0.15))

    assert float(expanded[62, 50, 0]) > float(tight[62, 50, 0]) + 0.15
    assert float(expanded[70, 50, 0]) < 0.08


def test_enhanced_blend_config_matches_opentalking_main_defaults() -> None:
    config = BlendConfig()

    assert config.mouth_dilation == 1.35
    assert config.feather == 1.15
    assert config.corner_expand == 0.12
    assert config.upper_margin == 0.85
    assert config.horizontal_margin == 0.22
    assert config.lower_lip_dynamic_expand == 0.25
    assert config.enable_jaw_motion_blend is False
    assert config.jaw_blend_alpha == 0.22


def test_jaw_motion_mask_adds_low_alpha_motion_below_mouth_only() -> None:
    geometry = MouthGeometry(
        center=(50, 50),
        rx=18,
        ry=7,
        outer_lip=((32, 50), (41, 45), (50, 44), (59, 45), (68, 50), (59, 55), (50, 57), (41, 55)),
    )
    config = BlendConfig(jaw_blend_alpha=0.25)
    mouth_mask = build_mouth_blend_mask((100, 100), geometry, config)

    jaw_mask = build_jaw_motion_mask((100, 100), geometry, mouth_mask, config)

    assert jaw_mask.shape == (100, 100, 1)
    assert 0.05 < float(jaw_mask[66, 50, 0]) <= 0.25
    assert float(jaw_mask[50, 50, 0]) < 0.02
    assert float(jaw_mask[88, 50, 0]) < 0.02


def test_enhanced_blend_preserves_pixels_outside_mouth() -> None:
    original = np.full((80, 80, 3), [120, 130, 140], dtype=np.uint8)
    pred = np.full((80, 80, 3), [40, 80, 180], dtype=np.uint8)
    geometry = MouthGeometry.ellipse(center=(40, 42), rx=14, ry=5)

    blended = blend_mouth_patch(pred, original, geometry=geometry, config=BlendConfig(color_match_strength=1.0))

    assert np.allclose(blended[39:46, 28:52].mean(axis=(0, 1)), [120, 130, 140], atol=8.0)
    assert np.array_equal(blended[64:72, 28:52], original[64:72, 28:52])


def test_jaw_motion_blend_is_controlled_independently_from_mouth_blend() -> None:
    original = np.full((100, 100, 3), [120, 130, 140], dtype=np.uint8)
    pred = np.full((100, 100, 3), [40, 80, 180], dtype=np.uint8)
    geometry = MouthGeometry.ellipse(center=(50, 50), rx=18, ry=6)

    without_jaw = blend_mouth_patch(
        pred,
        original,
        geometry=geometry,
        config=BlendConfig(color_match_strength=0.0, enable_jaw_motion_blend=False),
    )
    with_jaw = blend_mouth_patch(
        pred,
        original,
        geometry=geometry,
        config=BlendConfig(color_match_strength=0.0, enable_jaw_motion_blend=True, jaw_blend_alpha=0.25),
    )

    assert np.array_equal(without_jaw[66, 50], original[66, 50])
    assert not np.array_equal(with_jaw[66, 50], original[66, 50])
    assert np.array_equal(with_jaw[88, 50], original[88, 50])


def test_metadata_face_box_maps_to_frame_crop() -> None:
    metadata = {"face_box": [0.25, 0.1, 0.75, 0.9]}

    assert metadata_face_box_to_crop(metadata, (1000, 800)) == (100, 900, 200, 600)


def test_metadata_radius_maps_from_full_frame_to_input_crop() -> None:
    radius = metadata_radius_to_input_crop(
        normalized_radius=0.05,
        frame_size=720,
        crop_size=180,
        input_size=384,
    )

    assert radius == 77


def test_resize_reference_frame_uses_target_video_size() -> None:
    frame = np.zeros((2695, 1554, 3), dtype=np.uint8)

    resized = resize_reference_frame(frame, width=416, height=704)

    assert resized.shape == (704, 416, 3)


def test_basic_blend_does_not_paste_entire_face_crop() -> None:
    original = np.full((100, 100, 3), [120, 130, 140], dtype=np.uint8)
    pred = np.full((100, 100, 3), [10, 20, 30], dtype=np.uint8)

    blended = blend_mouth_patch_basic(pred, original)

    assert np.array_equal(blended[5, 5], original[5, 5])
    assert not np.array_equal(blended[65, 50], original[65, 50])


def test_enhanced_model_crop_stays_on_detector_crop() -> None:
    detector_crop = (363, 582, 271, 427)
    metadata_crop = (142, 771, 138, 563)

    assert select_wav2lip_model_crop(
        detector_crop=detector_crop,
        metadata_crop=metadata_crop,
        use_opentalking_improved=True,
    ) == detector_crop
