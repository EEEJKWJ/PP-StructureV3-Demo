import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.datasets.fintabnet_c_structure import (
    FinTabNetCStructureDataset,
)
from app.datasets.registry import DatasetRegistry
from app.datasets.preprocessing import (
    cells_to_structured_html,
    parse_structure_xml,
    prepare_split,
)


def object_xml(name, bbox):
    return f"""
    <object>
      <name>{name}</name>
      <bndbox>
        <xmin>{bbox[0]}</xmin><ymin>{bbox[1]}</ymin>
        <xmax>{bbox[2]}</xmax><ymax>{bbox[3]}</ymax>
      </bndbox>
    </object>
    """


def write_sample(root, stem="sample", spanning=True):
    test_dir = root / "test"
    image_dir = root / "images"
    words_dir = root / "words"
    test_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    words_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 40), "white").save(
        image_dir / f"{stem}.jpg"
    )
    objects = [
        object_xml("table", [0, 0, 100, 40]),
        object_xml("table projected row header", [0, 0, 100, 20]),
        object_xml("table column header", [0, 0, 100, 20]),
        object_xml("table row", [0, 0, 100, 20]),
        object_xml("table row", [0, 20, 100, 40]),
        object_xml("table column", [0, 0, 50, 40]),
        object_xml("table column", [50, 0, 100, 40]),
    ]
    if spanning:
        objects.append(
            object_xml("table spanning cell", [0, 0, 100, 20])
        )
    xml = (
        '<?xml version="1.0"?><annotation>'
        f"<size><width>100</width><height>40</height>"
        "<depth>3</depth></size>"
        + "".join(objects)
        + "</annotation>"
    )
    xml_path = test_dir / f"{stem}.xml"
    xml_path.write_text(xml, encoding="utf-8")
    words = [
        {"text": "Header", "bbox": [10, 5, 90, 15]},
        {"text": "Left", "bbox": [5, 25, 30, 35]},
        {"text": "Right", "bbox": [55, 25, 90, 35]},
    ]
    (words_dir / f"{stem}_words.json").write_text(
        json.dumps(words),
        encoding="utf-8",
    )
    return xml_path


class FinTabNetCPreprocessingTests(unittest.TestCase):
    def test_xml_to_cells_spans_text_and_html(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            xml_path = write_sample(root)
            result = parse_structure_xml(
                xml_path,
                root / "words" / "sample_words.json",
            )

        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["col_count"], 2)
        self.assertEqual(len(result["cells"]), 3)
        self.assertEqual(result["cells"][0]["colspan"], 2)
        self.assertEqual(result["cells"][0]["text"], "Header")
        self.assertIn('colspan="2"', result["html"])
        self.assertIn("Left", result["html"])
        self.assertEqual(result["warnings"], [])

    def test_no_spanning_cell_produces_full_grid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            xml_path = write_sample(root, spanning=False)
            result = parse_structure_xml(xml_path)

        self.assertEqual(len(result["cells"]), 4)
        self.assertTrue(
            all(cell["rowspan"] == 1 for cell in result["cells"])
        )

    def test_invalid_span_shape_is_warned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            xml_path = write_sample(root)
            text = xml_path.read_text(encoding="utf-8")
            span_start = text.index(
                "<name>table spanning cell</name>"
            )
            prefix = text[:span_start]
            suffix = text[span_start:]
            suffix = suffix.replace(
                "<xmin>0</xmin><ymin>0</ymin>",
                "<xmin>110</xmin><ymin>0</ymin>",
                1,
            ).replace(
                "<xmax>100</xmax><ymax>20</ymax>",
                "<xmax>170</xmax><ymax>20</ymax>",
                1,
            )
            xml_path.write_text(prefix + suffix, encoding="utf-8")

            result = parse_structure_xml(xml_path)

        self.assertTrue(result["warnings"])

    def test_html_escapes_cell_text_and_preserves_newlines(self):
        html = cells_to_structured_html([{
            "row": 0,
            "col": 0,
            "rowspan": 1,
            "colspan": 1,
            "text": "<A>\nB",
            "bbox": [0, 0, 1, 1],
        }])

        self.assertIn("&lt;A&gt;<br>B", html)

    def test_prepare_split_and_dataset_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            write_sample(root)
            cache = Path(temp_dir) / "cache" / "test.jsonl"
            metadata = prepare_split(root, "test", cache)
            dataset = FinTabNetCStructureDataset(
                root,
                split="test",
                cache_path=cache,
            )
            sample = next(dataset.iter_samples())

        self.assertEqual(metadata["written"], 1)
        self.assertEqual(dataset.total_samples, 1)
        self.assertEqual(sample.ground_truth["status"], "available")
        self.assertEqual(len(sample.ground_truth["cells"]), 3)
        self.assertIn("<table>", sample.ground_truth["html"])

    def test_prepare_failure_does_not_stop_other_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            write_sample(root, "good")
            bad = root / "test" / "bad.xml"
            bad.write_text("<annotation></annotation>", encoding="utf-8")
            cache = Path(temp_dir) / "cache" / "test.jsonl"

            metadata = prepare_split(root, "test", cache)

        self.assertEqual(metadata["written"], 1)
        self.assertEqual(len(metadata["failures"]), 1)

    def test_registry_exposes_structure_test_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            write_sample(root)
            cache = Path(temp_dir) / "cache" / "test.jsonl"
            prepare_split(root, "test", cache)
            registry = DatasetRegistry(
                fintabnet_path=root / "missing-pubtables",
                fintabnet_c_path=root,
            )

            dataset = registry.get_dataset(
                "fintabnet-c-structure-test"
            )

        self.assertEqual(
            dataset.describe()["type"],
            "fintabnet-c-structure",
        )

    def test_existing_cache_is_reused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            write_sample(root)
            cache = Path(temp_dir) / "cache" / "test.jsonl"
            first = prepare_split(root, "test", cache)
            second = prepare_split(root, "test", cache)

        self.assertEqual(first["written"], 1)
        self.assertEqual(second["written"], 1)


if __name__ == "__main__":
    unittest.main()
