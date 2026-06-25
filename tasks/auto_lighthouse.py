"""自动灯塔任务：扫描地图图标并按类型执行。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from loguru import logger

from core.adb_client import AdbClient
from core.deploy_march import DeployMarchHelper, StaminaInsufficientError
import cv2

from core.lighthouse_vision import (
    LIGHTHOUSE_MATCH_THRESHOLD,
    LIGHTHOUSE_SCAN_ROI,
    LighthouseMission,
    LighthouseScanResult,
    SKIP_MISSION_KINDS,
    classify_scanned_missions,
    is_lighthouse_intel_screen,
    refine_mission_click,
    scan_mission_icons,
)
from core.navigation import return_to_main_screen
from core.vision import MatchResult, Vision

StatusCallback = Callable[[str], None]

TEMPLATE_DIR = Path(__file__).parent.parent / "assets" / "templates"

BTN_TOWN_LABEL = "btn_town_label.png"
BTN_WILDERNESS_LABEL = "btn_wilderness_label.png"
SCENE_TOGGLE_ROI = (500, 1150, 720, 1280)
UI_CLEANUP_ATTEMPTS = 4

HERO_JOURNEY_WAIT_SEC = 5.0
MISSION_CONFIRM_PRE_WAIT_SEC = 1.0
MISSION_ICON_WAIT_SEC = 2.0
DEFAULT_INTERVAL = 3600.0
DEFAULT_STEP_DELAY = 1.5
DEFAULT_MONSTER_COOLDOWN = 120.0

MISSION_SCAN_SETTLE_SEC = 0.5
SCAN_EMPTY_RETRY_WAIT_SEC = 1.0
SCAN_REOPEN_WAIT_SEC = 1.2
MISSION_SLOT_MAX_ATTEMPTS = 3
MISSION_SLOT_TRACK_RADIUS = 28

DEFAULT_COORDS: dict[str, list[int]] = {
    "lighthouse_open": [666, 862],
    "hide_completed": [364, 1124],
    "mission_confirm_1": [354, 908],
    "mission_confirm_2": [350, 630],
    "hero_journey_finish": [528, 1196],
    "march": [560, 1200],
    "dialog_cancel": [250, 780],
}


def merge_task_config(cfg: dict) -> dict:
    coords = {**cfg.get("coords", {}), **DEFAULT_COORDS}
    return {
        "step_delay": cfg.get("step_delay", DEFAULT_STEP_DELAY),
        "coords": coords,
        "match_threshold": float(cfg.get("match_threshold", LIGHTHOUSE_MATCH_THRESHOLD)),
        "monster_cooldown": float(cfg.get("monster_cooldown", DEFAULT_MONSTER_COOLDOWN)),
    }


class AutoLighthouseTask:
    """野外 → 灯塔任务页 → 扫描并处理英雄之旅 / 帐篷 / 小怪。"""

    def __init__(
        self,
        adb: AdbClient,
        coords: dict[str, list[int]] | None = None,
        interval: float = DEFAULT_INTERVAL,
        formation_slot: int = 7,
        use_stamina: bool = True,
        step_delay: float = DEFAULT_STEP_DELAY,
        match_threshold: float = LIGHTHOUSE_MATCH_THRESHOLD,
        monster_cooldown: float = DEFAULT_MONSTER_COOLDOWN,
        on_status: StatusCallback | None = None,
    ):
        merged = merge_task_config(
            {
                "coords": coords or {},
                "step_delay": step_delay,
                "match_threshold": match_threshold,
                "monster_cooldown": monster_cooldown,
            }
        )
        self.adb = adb
        self.coords = merged["coords"]
        self.interval = interval
        self.formation_slot = formation_slot
        self.use_stamina = use_stamina
        self.step_delay = merged["step_delay"]
        self.match_threshold = merged["match_threshold"]
        self.monster_cooldown = merged["monster_cooldown"]
        self.on_status = on_status
        self._last_run = 0.0
        self._last_monster_dispatch_at = 0.0
        self._stop_event = threading.Event()
        self._slot_attempts: list[tuple[int, int, int]] = []
        self._beast_skip_centers: list[tuple[int, int]] = []
        self.vision = Vision(TEMPLATE_DIR, threshold=0.70)
        self._deploy = DeployMarchHelper(
            adb,
            formation_slot=formation_slot,
            use_stamina=use_stamina,
            coords=self.coords,
            on_status=on_status,
            interrupted=self._interrupted,
            step_delay=self.step_delay,
        )

    @property
    def name(self) -> str:
        return "自动灯塔任务"

    def stop(self) -> None:
        self._stop_event.set()
        self._emit("正在停止…")

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def _emit(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
        if self.on_status:
            self.on_status(message)

    def _interrupted(self) -> bool:
        return self._stop_event.is_set()

    def should_run(self) -> bool:
        return time.time() - self._last_run >= self.interval

    def _tap_xy(self, x: int, y: int, delay: float | None = None) -> None:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        self.adb.tap(x, y)
        time.sleep(delay if delay is not None else self.step_delay)

    def _tap(self, key: str, delay: float | None = None) -> None:
        x, y = self.coords[key]
        logger.debug(f"[{self.name}] 点击 {key} ({x}, {y})")
        self._tap_xy(x, y, delay)

    def _back(self, times: int = 1) -> None:
        for _ in range(times):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self.adb.back()
            time.sleep(0.5)

    def _dismiss_exit_dialog(self) -> None:
        """关闭「退出游戏」确认弹窗（点「返回」/取消，勿继续按 back）。"""
        if "dialog_cancel" in self.coords:
            self.adb.tap(*self.coords["dialog_cancel"])
            time.sleep(0.5)

    def _match_in_roi(
        self, screen, template: str, roi: tuple[int, int, int, int]
    ) -> MatchResult:
        x1, y1, x2, y2 = roi
        result = self.vision.match_template(screen[y1:y2, x1:x2], template)
        if not result.found:
            return result
        cx, cy = result.center
        return MatchResult(
            found=True,
            confidence=result.confidence,
            center=(x1 + cx, y1 + cy),
            top_left=(x1 + result.top_left[0], y1 + result.top_left[1]),
            size=result.size,
        )

    def _is_in_wilderness(self, screen) -> bool:
        return self._match_in_roi(screen, BTN_TOWN_LABEL, SCENE_TOGGLE_ROI).found

    def _is_in_town(self, screen) -> bool:
        return self._match_in_roi(screen, BTN_WILDERNESS_LABEL, SCENE_TOGGLE_ROI).found

    def _ensure_clean_ui(self) -> None:
        self._emit("清理界面…")
        for attempt in range(UI_CLEANUP_ATTEMPTS):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self._dismiss_exit_dialog()
            screen = self.adb.screenshot()
            if self._is_on_main_map(screen):
                self._emit("界面已就绪")
                time.sleep(0.5)
                return
            self._back(1)
            time.sleep(0.5)
        self._emit("已尝试清理界面，继续执行")

    def _is_on_main_map(self, screen) -> bool:
        return self._is_in_wilderness(screen) or self._is_in_town(screen)

    def _is_on_lighthouse_intel_page(self, screen) -> bool:
        """是否在灯塔情报页（区别于野外大地图）。"""
        if self._is_on_main_map(screen):
            return False
        if self._deploy.is_deploy_screen(screen):
            return False
        return is_lighthouse_intel_screen(screen)

    def _ensure_lighthouse_page(self) -> None:
        """确保当前在情报页；若在野外/出征界面则重新打开。"""
        for attempt in range(3):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self._dismiss_exit_dialog()
            screen = self.adb.screenshot()
            if self._is_on_lighthouse_intel_page(screen):
                return
            if self._deploy.is_deploy_screen(screen):
                self._emit("当前在出征界面，返回并重新打开情报页")
                self._back(2)
                time.sleep(0.8)
                continue
            if self._is_on_main_map(screen):
                self._emit("当前在野外大地图，重新打开情报页")
                self._open_lighthouse_page()
                return
            self._back(1)
            time.sleep(0.5)
        self._emit("未能确认情报页，尝试从野外重新打开")
        self._return_to_wilderness_main()
        self._open_lighthouse_page()

    def _ensure_wilderness(self) -> None:
        self._ensure_clean_ui()
        for attempt in range(10):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            self._dismiss_exit_dialog()
            screen = self.adb.screenshot()
            if self._is_in_wilderness(screen):
                self._emit("已在野外")
                return
            if self._is_in_town(screen):
                self._emit("当前在城镇，切换到野外…")
                result = self._match_in_roi(screen, BTN_WILDERNESS_LABEL, SCENE_TOGGLE_ROI)
                if result.found:
                    self._tap_xy(*result.center, delay=2.0)
                else:
                    self._back(1)
                continue
            self._back(1)
            time.sleep(0.5)
        raise RuntimeError("无法进入野外")

    def _return_to_wilderness_main(self) -> None:
        self._emit("返回野外主界面…")
        self._dismiss_exit_dialog()
        screen = self.adb.screenshot()
        if self._is_in_wilderness(screen):
            self._emit("已在野外")
            return
        return_to_main_screen(self.adb, on_status=self.on_status)
        self._dismiss_exit_dialog()
        self._ensure_wilderness()

    def _cancel_deploy_for_cooldown_skip(self) -> None:
        """小怪冷却跳过时：退出出征界面，尽量留在情报页继续扫描。"""
        self._dismiss_exit_dialog()
        for attempt in range(5):
            if self._interrupted():
                raise InterruptedError("任务已停止")
            screen = self.adb.screenshot()
            self._dismiss_exit_dialog()
            if self._is_on_lighthouse_intel_page(screen):
                self._emit("已回到情报页，继续扫描")
                return
            if self._is_on_main_map(screen):
                self._emit("已在野外，下一轮将重新打开情报页")
                return
            if self._deploy.is_deploy_screen(screen):
                self._emit("退出出征界面…")
                self._back(1)
                time.sleep(0.6)
                continue
            self._back(1)
            time.sleep(0.5)
        self._emit("未能确认界面，已尝试轻量返回")
        self._dismiss_exit_dialog()

    def _open_lighthouse_page(self) -> None:
        self._ensure_wilderness()
        self._emit("打开灯塔任务页面")
        self._tap("lighthouse_open", delay=2.0)
        self._prepare_lighthouse_scan()

    def _prepare_lighthouse_scan(self) -> None:
        self._emit("隐藏已完成任务")
        self._tap("hide_completed", delay=1.0)
        self._tap("hide_completed", delay=1.0)

    def _monster_cooldown_remaining(self) -> float:
        if self._last_monster_dispatch_at <= 0:
            return 0.0
        elapsed = time.time() - self._last_monster_dispatch_at
        return max(0.0, self.monster_cooldown - elapsed)

    def _sleep_interruptible(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._interrupted():
                raise InterruptedError("任务已停止")
            time.sleep(min(1.0, deadline - time.time()))

    def _format_missions_summary(self, missions: tuple[LighthouseMission, ...]) -> str:
        if not missions:
            return "无"
        return "，".join(
            f"{m.label}({m.center[0]},{m.center[1]},{m.confidence:.2f})"
            for m in missions
        )

    def _slot_attempt_count(self, center: tuple[int, int]) -> int:
        x, y = center
        for sx, sy, count in self._slot_attempts:
            if abs(x - sx) < MISSION_SLOT_TRACK_RADIUS and abs(y - sy) < MISSION_SLOT_TRACK_RADIUS:
                return count
        return 0

    def _is_slot_exhausted(self, center: tuple[int, int]) -> bool:
        return self._slot_attempt_count(center) >= MISSION_SLOT_MAX_ATTEMPTS

    def _record_slot_attempt(self, center: tuple[int, int]) -> int:
        """记录某坐标槽位的点击次数，返回当前次数。"""
        x, y = center
        for i, (sx, sy, count) in enumerate(self._slot_attempts):
            if abs(x - sx) < MISSION_SLOT_TRACK_RADIUS and abs(y - sy) < MISSION_SLOT_TRACK_RADIUS:
                count += 1
                self._slot_attempts[i] = (sx, sy, count)
                return count
        self._slot_attempts.append((x, y, 1))
        return 1

    def _is_beast_skip_center(self, center: tuple[int, int]) -> bool:
        x, y = center
        for sx, sy in self._beast_skip_centers:
            if abs(x - sx) < MISSION_SLOT_TRACK_RADIUS and abs(y - sy) < MISSION_SLOT_TRACK_RADIUS:
                return True
        return False

    def _mark_beast_skip_center(self, center: tuple[int, int]) -> None:
        if self._is_beast_skip_center(center):
            return
        self._beast_skip_centers.append(center)

    def _pick_executable_mission(
        self, missions: tuple[LighthouseMission, ...]
    ) -> LighthouseMission | None:
        """从扫描结果中取出下一个可点击的图标（跳过特殊大怪与已达尝试上限的坐标）。"""
        if not missions:
            return None
        for mission in missions:
            if mission.kind in SKIP_MISSION_KINDS:
                logger.debug(
                    f"[{self.name}] 跳过特殊大怪 "
                    f"({mission.center[0]},{mission.center[1]})"
                )
                continue
            if self._is_beast_skip_center(mission.center):
                logger.debug(
                    f"[{self.name}] 跳过已标记的特殊大怪坐标 "
                    f"({mission.center[0]},{mission.center[1]})"
                )
                continue
            if self._is_slot_exhausted(mission.center):
                logger.debug(
                    f"[{self.name}] 跳过已达上限坐标 "
                    f"({mission.center[0]},{mission.center[1]}) "
                    f"{self._slot_attempt_count(mission.center)}/{MISSION_SLOT_MAX_ATTEMPTS}"
                )
                continue
            return mission
        return None

    def _only_skippable_missions_remain(
        self, missions: tuple[LighthouseMission, ...]
    ) -> bool:
        if not missions:
            return False
        return all(
            m.kind in SKIP_MISSION_KINDS
            or self._is_beast_skip_center(m.center)
            or self._is_slot_exhausted(m.center)
            for m in missions
        )

    def _scan_missions(self, screen):
        if self._interrupted():
            raise InterruptedError("任务已停止")
        result = scan_mission_icons(
            screen,
            interrupted=self._interrupted,
        )
        if self._interrupted():
            raise InterruptedError("任务已停止")
        return result

    def _capture_and_scan(self) -> tuple[object, object]:
        if self._interrupted():
            raise InterruptedError("任务已停止")
        screen = self.adb.screenshot()
        self._emit("扫描任务图标…")
        result = self._scan_missions(screen)
        if result.missions:
            missions = classify_scanned_missions(screen, result.missions)
            result = LighthouseScanResult(
                mission=missions[0],
                missions=missions,
                best_confidence=result.best_confidence,
                best_label=missions[0].label,
                candidate_locations=result.candidate_locations,
            )
            logger.debug(
                f"[{self.name}] 扫描结果: {self._format_missions_summary(result.missions)}"
            )
        return screen, result

    def _find_executable_mission(self) -> tuple[object | None, object, LighthouseMission | None]:
        """扫描图标位置；仅在完全无候选时才刷新页面重试。"""
        self._sleep_interruptible(MISSION_SCAN_SETTLE_SEC)
        screen, result = self._capture_and_scan()
        mission = self._pick_executable_mission(result.missions)
        if mission is not None:
            return screen, result, mission

        if result.candidate_locations > 0:
            return screen, result, None

        self._emit(
            f"未发现任务图标，等待 {SCAN_EMPTY_RETRY_WAIT_SEC:.1f} 秒后重试…"
        )
        self._sleep_interruptible(SCAN_EMPTY_RETRY_WAIT_SEC)
        self._emit("刷新情报页筛选后重试…")
        self._prepare_lighthouse_scan()
        self._sleep_interruptible(0.8)
        screen, result = self._capture_and_scan()
        mission = self._pick_executable_mission(result.missions)
        if mission is not None or result.candidate_locations > 0:
            return screen, result, mission

        self._emit("再次刷新情报页筛选后重试…")
        self._prepare_lighthouse_scan()
        self._sleep_interruptible(SCAN_REOPEN_WAIT_SEC)
        screen, result = self._capture_and_scan()
        mission = self._pick_executable_mission(result.missions)
        return screen, result, mission

    def _log_scan_miss(self, result, screen) -> None:
        x1, y1, x2, y2 = LIGHTHOUSE_SCAN_ROI
        debug_dir = TEMPLATE_DIR.parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "lighthouse_scan_fail_roi.png"), screen[y1:y2, x1:x2])
        if result.candidate_locations > 0:
            self._emit(
                f"发现 {result.candidate_locations} 个颜色图钉候选，"
                f"但轮廓过滤后未能确认可点击图标"
            )
        else:
            self._emit("未发现任何任务图标")

    def _handle_stamina_after_action(self) -> bool:
        """处理体力弹窗。返回 True 表示已自动使用体力。"""
        try:
            return self._deploy.handle_stamina_popup_if_any()
        except StaminaInsufficientError:
            raise

    def _before_next_mission(self) -> None:
        """执行下一个小任务前：野外 → 打开灯塔情报页。"""
        self._dismiss_exit_dialog()
        screen = self.adb.screenshot()
        if self._is_on_lighthouse_intel_page(screen):
            self._emit("已在情报页，准备扫描")
            self._prepare_lighthouse_scan()
            return
        self._emit("准备下一个小任务，打开情报页")
        self._open_lighthouse_page()

    def _after_mission_completed(self) -> None:
        """每个小任务结束后：回到野外主界面。"""
        self._emit("小任务完成，返回野外")
        self._return_to_wilderness_main()

    def _tap_mission_confirms(self) -> None:
        """帐篷 / 英雄之旅 / 小怪：确认 1、确认 2 流程一致。"""
        time.sleep(MISSION_CONFIRM_PRE_WAIT_SEC)
        x1, y1 = self.coords["mission_confirm_1"]
        self._emit(f"点击确认 1 @ ({x1},{y1})")
        self._tap_xy(x1, y1, delay=1.5)
        x2, y2 = self.coords["mission_confirm_2"]
        self._emit(f"点击确认 2 @ ({x2},{y2})")
        self._tap_xy(x2, y2, delay=1.5)

    def _classify_post_click(self) -> str:
        """点击图标并确认弹窗后，根据当前界面判断任务类型。

        返回值: "small_monster" | "hero_journey" | "tent" | "wilderness"
        """
        time.sleep(0.5)
        screen = self.adb.screenshot()

        if self._deploy.is_deploy_screen(screen):
            self._emit("点击后识别为：需出征的任务（出征界面，具体类型见扫描分类）")
            return "small_monster"

        if self._is_on_lighthouse_intel_page(screen):
            self._emit("点击后识别为：帐篷（仍在情报页）")
            return "tent"

        if self._is_on_main_map(screen):
            self._emit("点击后识别为：已回到野外大地图")
            return "wilderness"

        self._emit("点击后识别为：英雄之旅（奖励界面）")
        return "hero_journey"

    def _click_icon_and_confirm(self, mission: LighthouseMission) -> None:
        """点击图标，等待弹窗，点击两次确认。"""
        self._emit(f"点击图标 @ ({mission.center[0]},{mission.center[1]})")
        self._tap_xy(*mission.center, delay=MISSION_ICON_WAIT_SEC)
        self._tap_mission_confirms()

    def _handle_post_confirm(self, kind: str) -> None:
        """确认弹窗后执行类型特定操作（导航由主循环统一处理）。"""
        if kind in ("tent", "wilderness"):
            self._emit("帐篷任务完成")
            return

        if kind == "hero_journey":
            self._emit("执行英雄之旅后续…")
            self._tap("hero_journey_finish", delay=1.0)
            self._emit(f"等待 {int(HERO_JOURNEY_WAIT_SEC)} 秒…")
            time.sleep(HERO_JOURNEY_WAIT_SEC)
            self._handle_stamina_after_action()
            return

        if kind == "small_monster":
            self._emit("执行小怪出征…")
            self._deploy.select_formation()
            self._deploy.tap_march()
            return

        raise RuntimeError(f"未知任务类型：{kind}")

    def _execute_tent(self, mission: LighthouseMission) -> bool:
        self._emit(f"处理帐篷 @ {mission.center}")
        self._tap_xy(*mission.center, delay=MISSION_ICON_WAIT_SEC)
        self._tap_mission_confirms()
        self._emit("帐篷任务完成")
        return True

    def _execute_hero_journey(self, mission: LighthouseMission) -> bool:
        self._emit(f"处理英雄之旅 @ {mission.center}")
        self._tap_xy(*mission.center, delay=MISSION_ICON_WAIT_SEC)
        self._tap_mission_confirms()
        self._tap("hero_journey_finish", delay=1.0)
        self._emit(f"等待 {int(HERO_JOURNEY_WAIT_SEC)} 秒…")
        time.sleep(HERO_JOURNEY_WAIT_SEC)
        self._handle_stamina_after_action()
        return True

    def _execute_small_monster(self, mission: LighthouseMission) -> bool:
        self._emit(f"处理小怪 @ {mission.center}")
        self._tap_xy(*mission.center, delay=MISSION_ICON_WAIT_SEC)
        self._tap_mission_confirms()
        try:
            self._deploy.select_formation()
            self._deploy.tap_march()
        except StaminaInsufficientError:
            raise
        self._return_to_wilderness_main()
        self._open_lighthouse_page()
        return True

    def _execute_mission(self, mission: LighthouseMission) -> bool:
        if mission.kind == "tent":
            return self._execute_tent(mission)
        if mission.kind == "hero_journey":
            return self._execute_hero_journey(mission)
        if mission.kind == "small_monster" or mission.kind == "small_monster_beast":
            return self._execute_small_monster(mission)
        raise RuntimeError(f"未知灯塔任务类型：{mission.kind}")

    def run_lighthouse_cycle(self) -> int:
        """扫描图标 → 点击 → 确认 → 分类 → 处理 → 回野外；下一轮再开情报页。"""
        handled = 0
        self._slot_attempts.clear()
        self._beast_skip_centers.clear()

        while not self._interrupted():
            self._before_next_mission()

            screen, result, mission = self._find_executable_mission()
            if self._interrupted():
                raise InterruptedError("任务已停止")

            if mission is None:
                if result.missions and self._only_skippable_missions_remain(
                    result.missions
                ):
                    beast_count = sum(
                        1 for m in result.missions if m.kind in SKIP_MISSION_KINDS
                    )
                    if beast_count:
                        self._emit(
                            f"剩余 {beast_count} 个特殊大怪任务已跳过，"
                            f"结束本轮"
                        )
                    else:
                        self._emit(
                            f"剩余 {len(result.missions)} 个候选坐标均已尝试 "
                            f"{MISSION_SLOT_MAX_ATTEMPTS} 次，结束本轮"
                        )
                else:
                    self._log_scan_miss(result, screen)
                    self._emit("页面上没有可处理的灯塔任务，扫描结束")
                self._back(1)
                self._after_mission_completed()
                break

            kind_hint = mission.label if mission.kind else "未分类"
            self._emit(
                f"发现{kind_hint} ({mission.center[0]},{mission.center[1]})，"
                f"共 {len(result.missions)} 个候选"
            )

            tap_target = refine_mission_click(mission, screen, result.missions)
            if tap_target.kind in SKIP_MISSION_KINDS:
                self._emit(
                    f"扫描识别为特殊大怪 ({tap_target.center[0]},{tap_target.center[1]})，"
                    f"跳过不打"
                )
                self._mark_beast_skip_center(tap_target.center)
                continue
            if tap_target.center != mission.center:
                self._emit(
                    f"相邻图标校正落点 -> ({tap_target.center[0]},{tap_target.center[1]})"
                )

            attempt_no = self._record_slot_attempt(tap_target.center)
            self._emit(
                f"坐标 ({tap_target.center[0]},{tap_target.center[1]}) "
                f"第 {attempt_no}/{MISSION_SLOT_MAX_ATTEMPTS} 次尝试"
            )
            self._click_icon_and_confirm(tap_target)
            kind = self._classify_post_click()

            if kind == "small_monster":
                if tap_target.kind in SKIP_MISSION_KINDS:
                    self._emit("点击后确认为特殊大怪，跳过（不打）")
                    self._mark_beast_skip_center(tap_target.center)
                    self._cancel_deploy_for_cooldown_skip()
                    continue
                cooldown = self._monster_cooldown_remaining()
                if cooldown > 0:
                    mins, secs = divmod(int(cooldown), 60)
                    self._emit(
                        f"识别为小怪但冷却中（剩余 {mins} 分 {secs} 秒），"
                        f"跳过此图标"
                    )
                    self._cancel_deploy_for_cooldown_skip()
                    continue
                self._last_monster_dispatch_at = time.time()

            try:
                self._handle_post_confirm(kind)
            except StaminaInsufficientError:
                raise
            except Exception as exc:
                self._emit(f"执行失败：{exc}")

            handled += 1
            self._after_mission_completed()

        return handled

    def run_once(self, *, force: bool = False) -> bool:
        if not force and not self.should_run():
            return False

        self._last_run = time.time()
        try:
            count = self.run_lighthouse_cycle()
            self._return_to_wilderness_main()
            self._emit(
                f"本轮处理 {count} 个灯塔任务，"
                f"{int(self.interval // 60)} 分钟后再次扫描"
            )
            return True
        except InterruptedError:
            self._emit("任务已停止")
            raise
        except StaminaInsufficientError as exc:
            self._emit(str(exc))
            self._return_to_wilderness_main()
            return False
        except Exception as exc:
            logger.exception(f"[{self.name}] 执行失败")
            self._emit(f"执行失败：{exc}")
            try:
                self._return_to_wilderness_main()
            except Exception as nav_exc:
                logger.warning(f"[{self.name}] 返回野外失败: {nav_exc}")
            return False
