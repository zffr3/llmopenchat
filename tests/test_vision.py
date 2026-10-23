from __future__ import annotations

import base64
import copy
import dataclasses
import json
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image, PngImagePlugin

from llmopenchat import vision
from llmopenchat.vision import (
    VisionError, VisionSettings, content_text, has_images, image_candidate,
    image_message, incoming_image, prepare_image, prune_images, validate_content,
)


def image_bytes(*, format="PNG", size=(32, 24), mode="RGB", color="red", **options):
    with Image.new(mode, size, color) as image, BytesIO() as output:
        image.save(output, format=format, **options)
        return output.getvalue()


def data_part(content: bytes, mime="png"):
    return {"type": "image_url", "image_url": {
        "url": f"data:image/{mime};base64," + base64.b64encode(content).decode("ascii"),
    }}


def part_bytes(part):
    return base64.b64decode(part["image_url"]["url"].split(",", 1)[1], validate=True)


class VisionSettingsTests(unittest.TestCase):
    def test_defaults_are_disabled_and_bounded(self):
        settings = VisionSettings.from_config({})
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.max_input_bytes, 10 * 1024 * 1024)
        self.assertEqual(settings.max_pixels, 24_000_000)
        self.assertEqual(settings.max_side, 1536)
        self.assertEqual(settings.max_image_bytes, 512 * 1024)
        self.assertEqual(settings.max_history_images, 4)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            settings.enabled = True

    def test_strict_types_unknown_options_and_hard_caps_are_rejected(self):
        bad = [None, [], True, {"enabled": 1}, {"enabled": "true"}, {"surprise": 1},
               {"max_input_bytes": 20 * 1024 * 1024 + 1}, {"max_pixels": 40_000_001},
               {"max_side": 2049}, {"max_image_bytes": 1024 * 1024 + 1},
               {"max_history_images": 5}, {"max_history_images": 0}, {"max_input_bytes": True},
               {"max_pixels": 1.0}, {"max_side": -1}, {"max_image_bytes": 0}]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(VisionError):
                VisionSettings.from_config({"vision": values})
        with self.assertRaises(VisionError):
            VisionSettings.from_config([])
        with self.assertRaises(VisionError):
            VisionSettings(max_side=10000)

    def test_supported_hard_maximums_are_accepted(self):
        settings = VisionSettings(enabled=True, max_input_bytes=vision.MAX_INPUT_BYTES,
                                  max_pixels=vision.MAX_PIXELS, max_side=vision.MAX_SIDE,
                                  max_image_bytes=vision.MAX_IMAGE_BYTES, max_history_images=4)
        self.assertTrue(settings.enabled)

    def test_text_helpers_and_config_do_not_require_pillow(self):
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None, "PIL.ImageOps": None}):
            self.assertFalse(VisionSettings.from_config({}).enabled)
            self.assertEqual(content_text("text only"), "text only")
            self.assertFalse(has_images("text only"))
            self.assertEqual(validate_content([{"type": "text", "text": "only"}]), [{"type": "text", "text": "only"}])
            with self.assertRaisesRegex(VisionError, "pip install"):
                prepare_image(b"image", VisionSettings(enabled=True))


class ImageSelectionTests(unittest.TestCase):
    def test_only_photo_or_supported_image_document_is_candidate(self):
        for kind, filename, mime in (("photo", "", ""), ("document", "PIC.JPEG", "application/octet-stream"),
                                     ("document", "file", "IMAGE/PNG; charset=binary"), ("document", "file.webp", "")):
            self.assertTrue(image_candidate(kind, filename, mime))
        for kind, filename, mime in (("document", "file.pdf", "application/pdf"), ("document", "animation.gif", "image/gif"),
                                     ("sticker", "file.webp", "image/webp"), ("animation", "file.png", "image/png"),
                                     ("video", "file.jpg", "image/jpeg")):
            self.assertFalse(image_candidate(kind, filename, mime))

    def test_largest_photo_is_selected_by_area_then_file_size(self):
        small = {"file_id": "small", "width": 100, "height": 100, "file_size": 10000}
        large = {"file_id": "large", "width": 200, "height": 100, "file_size": 20}
        largest = {**large, "file_id": "winner", "file_size": 50}
        message = {"photo": [None, {}, small, large, largest]}
        selected = incoming_image(message)
        self.assertEqual(selected, {**largest, "media_type": "photo"})
        selected["file_id"] = "changed"
        self.assertEqual(largest["file_id"], "winner")

    def test_image_document_selected_without_sticker_animation_or_other_files(self):
        document = {"file_id": "picture", "file_name": "image.png", "mime_type": "application/octet-stream"}
        self.assertEqual(incoming_image({"document": document}), {**document, "media_type": "document"})
        for message in ({"animation": document}, {"sticker": document}, {"photo": []}, None,
                        {"animation": document, "document": document}, {"sticker": document, "document": document},
                        {"document": {**document, "file_name": "archive.zip"}}, {"document": {"file_name": "image.png"}}):
            self.assertIsNone(incoming_image(message))


class ImagePreparationTests(unittest.TestCase):
    def setUp(self):
        self.settings = VisionSettings(enabled=True)

    def test_jpeg_png_and_webp_are_decoded_and_normalized_to_jpeg(self):
        for format in ("JPEG", "PNG", "WEBP"):
            with self.subTest(format=format):
                result = prepare_image(image_bytes(format=format), self.settings)
                self.assertTrue(result["image_url"]["url"].startswith("data:image/jpeg;base64,"))
                self.assertLessEqual(len(part_bytes(result)), self.settings.max_image_bytes)
                with Image.open(BytesIO(part_bytes(result))) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (32, 24))
                self.assertEqual(validate_content([result]), [result])

    def test_orientation_is_applied_and_metadata_is_not_sent(self):
        exif = Image.Exif()
        exif[274] = 6
        exif[315] = "PRIVATE_ARTIST_METADATA"
        content = image_bytes(format="JPEG", size=(40, 20), exif=exif, icc_profile=b"PRIVATE_ICC_METADATA")
        normalized = part_bytes(prepare_image(content, self.settings))
        with Image.open(BytesIO(normalized)) as image:
            self.assertEqual(image.size, (20, 40))
            self.assertNotIn("exif", image.info)
            self.assertNotIn("icc_profile", image.info)
            self.assertFalse(image.getexif())
        self.assertNotIn(b"PRIVATE_ARTIST_METADATA", normalized)
        self.assertNotIn(b"PRIVATE_ICC_METADATA", normalized)
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("Comment", "PRIVATE_PNG_COMMENT")
        normalized = part_bytes(prepare_image(image_bytes(pnginfo=png_info), self.settings))
        self.assertNotIn(b"PRIVATE_PNG_COMMENT", normalized)

    def test_transparency_is_composited_on_white(self):
        content = image_bytes(mode="RGBA", color=(255, 0, 0, 0))
        with Image.open(BytesIO(part_bytes(prepare_image(content, self.settings)))) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertTrue(all(channel >= 250 for channel in image.getpixel((10, 10))))
        with Image.new("P", (10, 10), 0) as image, BytesIO() as output:
            image.putpalette([0, 0, 0] + [0] * 765)
            image.save(output, format="PNG", transparency=0)
            content = output.getvalue()
        with Image.open(BytesIO(part_bytes(prepare_image(content, self.settings)))) as image:
            self.assertTrue(all(channel >= 250 for channel in image.getpixel((5, 5))))

    def test_large_side_is_resized_and_small_output_limit_is_respected(self):
        result = prepare_image(image_bytes(size=(3000, 1000)), self.settings)
        with Image.open(BytesIO(part_bytes(result))) as image:
            self.assertEqual(image.size, (1536, 512))
        with Image.effect_noise((900, 900), 100).convert("RGB") as noise, BytesIO() as output:
            noise.save(output, format="PNG")
            content = output.getvalue()
        result = prepare_image(content, dataclasses.replace(self.settings, max_image_bytes=32 * 1024))
        self.assertLessEqual(len(part_bytes(result)), 32 * 1024)
        with Image.open(BytesIO(part_bytes(result))) as image:
            self.assertLess(max(image.size), 900)

    def test_invalid_wrong_format_truncated_animated_and_oversized_are_rejected(self):
        valid = image_bytes()
        cases = [b"", b"not an image", b"<svg xmlns='http://www.w3.org/2000/svg'/>",
                 image_bytes(format="GIF"), image_bytes(format="BMP"), valid[:len(valid) // 2]]
        for content in cases:
            with self.subTest(content=content[:20]), self.assertRaises(VisionError):
                prepare_image(content, self.settings)
        with self.assertRaises(VisionError):
            prepare_image(valid, dataclasses.replace(self.settings, max_input_bytes=len(valid) - 1))
        with self.assertRaises(VisionError):
            prepare_image(valid, dataclasses.replace(self.settings, max_pixels=100))
        with self.assertRaises(VisionError):
            prepare_image("not bytes", self.settings)
        for format in ("PNG", "WEBP"):
            with self.subTest(format=format), Image.new("RGB", (16, 16), "red") as first, Image.new("RGB", (16, 16), "blue") as second, BytesIO() as output:
                first.save(output, format=format, save_all=True, append_images=[second], duration=100, loop=0)
                with self.assertRaisesRegex(VisionError, "Анимированные"):
                    prepare_image(output.getvalue(), self.settings)

    def test_disabled_mode_does_not_silently_send_image(self):
        with self.assertRaisesRegex(VisionError, "отключена"):
            prepare_image(image_bytes(), VisionSettings())
        with self.assertRaises(VisionError):
            image_message("caption", image_bytes(), {})

    def test_tiny_output_limit_fails_instead_of_exceeding_it(self):
        with self.assertRaisesRegex(VisionError, "невозможно уменьшить"):
            prepare_image(image_bytes(size=(1, 1)), dataclasses.replace(self.settings, max_image_bytes=1))

    def test_message_has_standard_text_and_image_parts(self):
        content = image_message("Что на фото?", image_bytes(), {"vision": {"enabled": True}})
        self.assertEqual(content[0], {"type": "text", "text": "Что на фото?"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(validate_content(content), content)


class VisionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.image = data_part(image_bytes())
        self.text = {"type": "text", "text": "text"}

    def test_validated_content_is_detached_and_preserves_exact_image(self):
        original = [self.text, self.image]
        cleaned = validate_content(original)
        self.assertEqual(cleaned, original)
        cleaned[0]["text"] = "changed"
        cleaned[1]["image_url"]["url"] = "changed"
        self.assertEqual(original[0]["text"], "text")
        self.assertTrue(original[1]["image_url"]["url"].startswith("data:image/png"))

    def test_previews_and_image_detection_never_include_image_data(self):
        content = [self.text, self.image, {"type": "unknown", "url": "SECRET"}]
        self.assertEqual(content_text(content), "text\n[Изображение]")
        self.assertNotIn("base64", content_text(content))
        self.assertNotIn("SECRET", content_text(content))
        self.assertTrue(has_images(content))
        self.assertFalse(has_images("data:image/png;base64,not-a-part"))
        self.assertFalse(has_images([self.text, None]))
        self.assertEqual(content_text(None), "")
        self.assertEqual(content_text("plain"), "plain")

    def test_external_urls_paths_svg_bad_base64_and_wrong_mime_are_rejected(self):
        urls = ["https://example.com/image.jpg", "file:///secret.jpg", "C:\\secret.png", "data:image/svg+xml;base64,PHN2Zy8+",
                "data:image/png;base64,", "data:image/png;base64,@@@", "data:image/png;base64,Zh==",
                "data:image/png;base64, YQ==", "data:image/png;base64,YQ==\n"]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(VisionError):
                validate_content([{"type": "image_url", "image_url": {"url": url}}])
        with self.assertRaises(VisionError):
            validate_content([data_part(image_bytes(format="JPEG"), "png")])

    def test_unknown_part_fields_invalid_structure_and_part_image_caps_are_rejected(self):
        bad = [None, "text", [], [None], [{"type": "text", "text": 5}],
               [{"type": "text", "text": "valid", "url": "ignored"}],
               [{"type": "image_url", "image_url": {**self.image["image_url"], "detail": "high"}}],
               [{**self.image, "extra": "ignored"}], [{"type": "input_image", "image_url": self.image["image_url"]}],
               [self.text] * 33, [self.image] * 5]
        for content in bad:
            with self.subTest(content_type=type(content)), self.assertRaises(VisionError):
                validate_content(content)

    def test_saved_images_have_absolute_byte_and_dimension_bounds(self):
        oversized = "data:image/png;base64," + "A" * (4 * ((vision.MAX_IMAGE_BYTES + 2) // 3) + 100)
        with self.assertRaises(VisionError):
            validate_content([{"type": "image_url", "image_url": {"url": oversized}}])
        with self.assertRaises(VisionError):
            validate_content([data_part(image_bytes(size=(2049, 1)))])
        with Image.new("RGB", (16, 16), "red") as first, Image.new("RGB", (16, 16), "blue") as second, BytesIO() as output:
            first.save(output, format="PNG", save_all=True, append_images=[second], duration=100)
            with self.assertRaises(VisionError):
                validate_content([data_part(output.getvalue())])

    def test_pruning_keeps_latest_n_globally_and_leaves_placeholders(self):
        messages = [{"role": "user", "content": [copy.deepcopy(self.text), copy.deepcopy(self.image)]} for _ in range(5)]
        messages.insert(2, {"role": "assistant", "content": "answer"})
        prune_images(messages, {"vision": {"enabled": True, "max_history_images": 2}})
        self.assertEqual(sum(has_images(message["content"]) for message in messages), 2)
        self.assertEqual(messages[0]["content"][1], {"type": "text", "text": vision.REMOVED_IMAGE_PLACEHOLDER})
        self.assertTrue(has_images(messages[-1]["content"]))
        self.assertTrue(has_images(messages[-2]["content"]))
        self.assertEqual(messages[2]["content"], "answer")
        self.assertNotIn("base64", json.dumps(messages[:3]))

    def test_disabling_vision_does_not_destroy_image_history_and_nonuser_parts_are_removed(self):
        messages = [{"role": "user", "content": [copy.deepcopy(self.image)]},
                    {"role": "assistant", "content": [copy.deepcopy(self.image)]}]
        prune_images(messages, {"vision": {"enabled": False}})
        self.assertTrue(has_images(messages[0]["content"]))
        self.assertFalse(has_images(messages[1]["content"]))


if __name__ == "__main__":
    unittest.main()
