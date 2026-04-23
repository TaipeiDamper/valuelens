from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, Qt

RESIZE_MARGIN = 6
MIN_WIDTH = 320
MIN_HEIGHT = 160

EDGE_LEFT = 1
EDGE_RIGHT = 2
EDGE_TOP = 4
EDGE_BOTTOM = 8


@dataclass
class InteractionResult:
    move_to: QPoint | None = None
    resize_to: QRect | None = None
    cursor: Qt.CursorShape | None = None


class WindowInteractionController:
    def __init__(self) -> None:
        self.drag_pos: QPoint | None = None
        self.resize_edges = 0
        self.resize_start_geom: QRect | None = None
        self.resize_start_global: QPoint | None = None
        self.is_dragging = False
        self.is_resizing = False

    def edges_at(self, pos: QPoint, width: int, height: int) -> int:
        edges = 0
        if pos.x() <= RESIZE_MARGIN:
            edges |= EDGE_LEFT
        if pos.x() >= width - RESIZE_MARGIN:
            edges |= EDGE_RIGHT
        if pos.y() <= RESIZE_MARGIN:
            edges |= EDGE_TOP
        if pos.y() >= height - RESIZE_MARGIN:
            edges |= EDGE_BOTTOM
        return edges

    def cursor_for_edges(self, edges: int) -> Qt.CursorShape:
        if edges in (EDGE_LEFT | EDGE_TOP, EDGE_RIGHT | EDGE_BOTTOM):
            return Qt.CursorShape.SizeFDiagCursor
        if edges in (EDGE_RIGHT | EDGE_TOP, EDGE_LEFT | EDGE_BOTTOM):
            return Qt.CursorShape.SizeBDiagCursor
        if edges & (EDGE_LEFT | EDGE_RIGHT):
            return Qt.CursorShape.SizeHorCursor
        if edges & (EDGE_TOP | EDGE_BOTTOM):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def start_panel_drag(self, global_point: QPoint, frame_top_left: QPoint) -> None:
        self.is_dragging = True
        self.drag_pos = global_point - frame_top_left

    def handle_mouse_press(
        self,
        *,
        pos: QPoint,
        global_pos: QPoint,
        width: int,
        height: int,
        geometry: QRect,
    ) -> bool:
        edges = self.edges_at(pos, width, height)
        if edges:
            self.is_resizing = True
            self.resize_edges = edges
            self.resize_start_global = global_pos
            self.resize_start_geom = QRect(geometry)
            return True
        self.is_dragging = True
        self.drag_pos = global_pos - geometry.topLeft()
        return False

    def handle_mouse_move(
        self,
        *,
        pos: QPoint,
        global_pos: QPoint,
        left_pressed: bool,
        width: int,
        height: int,
    ) -> InteractionResult:
        if not left_pressed:
            return InteractionResult(cursor=self.cursor_for_edges(self.edges_at(pos, width, height)))

        if self.is_resizing and self.resize_start_geom is not None and self.resize_start_global is not None:
            delta = global_pos - self.resize_start_global
            g = QRect(self.resize_start_geom)
            if self.resize_edges & EDGE_LEFT:
                new_left = min(g.right() - MIN_WIDTH + 1, g.left() + delta.x())
                g.setLeft(new_left)
            if self.resize_edges & EDGE_RIGHT:
                g.setRight(max(g.left() + MIN_WIDTH - 1, g.right() + delta.x()))
            if self.resize_edges & EDGE_TOP:
                new_top = min(g.bottom() - MIN_HEIGHT + 1, g.top() + delta.y())
                g.setTop(new_top)
            if self.resize_edges & EDGE_BOTTOM:
                g.setBottom(max(g.top() + MIN_HEIGHT - 1, g.bottom() + delta.y()))
            return InteractionResult(resize_to=g)

        if self.drag_pos is not None:
            return InteractionResult(move_to=global_pos - self.drag_pos)

        return InteractionResult()

    def reset(self) -> None:
        self.drag_pos = None
        self.is_dragging = False
        self.is_resizing = False
        self.resize_edges = 0
        self.resize_start_geom = None
        self.resize_start_global = None

