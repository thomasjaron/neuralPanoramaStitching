from neural_stitch.config import OutputConfig
from neural_stitch.render import output_size


def test_preset_sets_short_side_exactly():
    assert output_size((4000, 1500), OutputConfig(resolution="1080p")) == ((2880, 1080), None)
    (w, h), _ = output_size((1500, 4000), OutputConfig(resolution="720p"))  # portrait
    assert (w, h) == (720, 1920)


def test_preset_never_upscales():
    size, note = output_size((1200, 800), OutputConfig(resolution="1080p"))
    assert size == (1200, 800)
    assert "larger than the images allow" in note


def test_native_respects_max_side():
    assert output_size((8000, 2000), OutputConfig(max_side=4000)) == ((4000, 1000), None)
    assert output_size((1000, 500), OutputConfig(max_side=4000)) == ((1000, 500), None)


def test_megapixel_limit():
    (w, h), note = output_size((40000, 8000), OutputConfig(resolution="4320p", max_megapixels=50))
    assert w * h <= 50.1e6 and "megapixels" in note
