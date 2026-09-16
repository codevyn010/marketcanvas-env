import pytest
from marketcanvas.canvas import Canvas
from marketcanvas.elements import ElementType, ShapeVariant


class TestCanvasBasics:
    def test_empty_canvas(self):
        c = Canvas()
        assert c.width == 800
        assert c.height == 600
        assert len(c.elements) == 0

    def test_add_text(self):
        c = Canvas()
        elem = c.add_text("Hello", x=10, y=20, width=100, height=30)
        assert elem.id == 1
        assert elem.element_type == ElementType.TEXT
        assert elem.content == "Hello"
        assert len(c.elements) == 1

    def test_add_shape(self):
        c = Canvas()
        elem = c.add_shape(x=0, y=0, width=50, height=50, variant=ShapeVariant.BUTTON)
        assert elem.shape_variant == ShapeVariant.BUTTON
        assert elem.id == 1

    def test_add_image(self):
        c = Canvas()
        elem = c.add_image(x=10, y=10, width=100, height=100, label="Logo")
        assert elem.element_type == ElementType.IMAGE
        assert elem.label == "Logo"

    def test_sequential_ids(self):
        c = Canvas()
        e1 = c.add_text("A", x=0, y=0, width=50, height=20)
        e2 = c.add_shape(x=0, y=0, width=50, height=20)
        e3 = c.add_image(x=0, y=0, width=50, height=20)
        assert (e1.id, e2.id, e3.id) == (1, 2, 3)

    def test_z_ordering(self):
        c = Canvas()
        c.add_text("Back", x=0, y=0, width=50, height=20)
        c.add_text("Front", x=0, y=0, width=50, height=20)
        elements = c.elements
        assert elements[0].z_index < elements[1].z_index


class TestCanvasMutations:
    def test_move_element(self):
        c = Canvas()
        elem = c.add_text("X", x=10, y=10, width=50, height=20)
        assert c.move_element(elem.id, 100, 200)
        assert elem.x == 100
        assert elem.y == 200

    def test_move_nonexistent(self):
        c = Canvas()
        assert c.move_element(999, 0, 0) is False

    def test_resize_element(self):
        c = Canvas()
        elem = c.add_shape(x=0, y=0, width=100, height=100)
        assert c.resize_element(elem.id, 200, 50)
        assert elem.width == 200
        assert elem.height == 50

    def test_remove_element(self):
        c = Canvas()
        elem = c.add_text("Gone", x=0, y=0, width=50, height=20)
        assert c.remove_element(elem.id)
        assert len(c.elements) == 0
        assert c.remove_element(elem.id) is False

    def test_update_element(self):
        c = Canvas()
        elem = c.add_text("Old", x=0, y=0, width=50, height=20)
        c.update_element(elem.id, content="New")
        assert elem.content == "New"

    def test_clear(self):
        c = Canvas()
        c.add_text("A", x=0, y=0, width=50, height=20)
        c.add_shape(x=0, y=0, width=50, height=20)
        c.clear()
        assert len(c.elements) == 0


class TestBoundsClamping:
    def test_clamp_position(self):
        c = Canvas(width=800, height=600)
        elem = c.add_text("X", x=900, y=700, width=50, height=20)
        assert elem.x <= c.width - 1
        assert elem.y <= c.height - 1

    def test_clamp_negative(self):
        c = Canvas()
        elem = c.add_text("X", x=-50, y=-50, width=100, height=50)
        assert elem.x == 0
        assert elem.y == 0

    def test_clamp_size(self):
        c = Canvas(width=800, height=600)
        elem = c.add_shape(x=750, y=550, width=200, height=200)
        assert elem.right <= c.width
        assert elem.bottom <= c.height


class TestHitTesting:
    def test_elements_at_point(self):
        c = Canvas()
        e1 = c.add_shape(x=0, y=0, width=100, height=100)
        e2 = c.add_shape(x=50, y=50, width=100, height=100)
        hits = c.elements_at(75, 75)
        assert len(hits) == 2
        assert hits[0].id == e2.id  # higher z-index first

    def test_no_hit(self):
        c = Canvas()
        c.add_shape(x=0, y=0, width=50, height=50)
        assert len(c.elements_at(200, 200)) == 0


class TestSerialization:
    def test_to_dict(self):
        c = Canvas()
        c.add_text("Hi", x=10, y=20, width=100, height=30)
        d = c.to_dict()
        assert d["canvas"]["width"] == 800
        assert d["canvas"]["element_count"] == 1
        assert d["elements"][0]["content"] == "Hi"
