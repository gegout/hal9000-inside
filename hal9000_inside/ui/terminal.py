"""Terminal UI for HAL 9000 Inside.

Renders colored state transitions:
[LISTENING] / [THINKING] / [SPEAKING] / [ERROR]
Displays:
- Provisional transcripts (live in-progress speech)
- Committed transcripts (final recognized turn)
- Spoken responses
- Device status and control indicators
"""

import sys
import time
from typing import Optional

# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
DIM = "\033[2m"


class TerminalUI:
    """Async/Console UI for displaying speech pipeline states and messages compactly."""

    # Emoticons for statuses
    STATUS_ICONS = {
        "LISTENING": "👂",       # Just listening
        "VOICE_DETECTED": "🗣️ ",  # User speaking / voice detected
        "THINKING": "🧠",        # Model thinking / transcribing
        "RESPONDING": "🔴",       # Model responding / speaking (HAL 9000 eye)
        "ERROR": "⚠️ ",          # Error
        "IDLE": "⏸️ ",           # Idle / paused
        "SLEEPING": "😴",       # Sleeping
        "MUTED": "🔇",          # Microphone muted
        "PTT": "🎙️ ",           # Push-to-Talk active
    }

    def __init__(self):
        self.current_state = "LISTENING"
        self._current_user_text = ""
        self._last_line_len = 0
        self._progress_active = False

    def header(self):
        print(f"{BOLD}{RED}======================================================{RESET}")
        print(f"{BOLD}{RED}           HAL 9000 INSIDE - VOICE ASSISTANT          {RESET}")
        print(f"{BOLD}{WHITE}  \"Good afternoon, gentlemen. I am a HAL 9000 computer.\" {RESET}")
        print(f"{BOLD}{RED}======================================================{RESET}")
        print(f"{DIM}Controls: [Space] Push-to-Talk | [M] Mute | [C] Cancel | [S] Sleep | [Q] Quit{RESET}\n")

    def make_gauge(self, signal: float, noise_floor: float, is_speaking: bool) -> str:
        """Create a compact mic raw input gauge showing noise vs signal power."""
        gauge_width = 8
        snr = max(0.0, signal - noise_floor)
        # Adaptively normalize scale (default max 350.0, higher if elevated noise floor)
        max_scale = max(noise_floor * 2.5, 350.0)
        filled = min(gauge_width, max(0, int((signal / max_scale) * gauge_width)))
        
        # Color based on whether voice is active or background noise
        if is_speaking:
            bar_color = GREEN
            bar_chars = "■" * filled + "·" * (gauge_width - filled)
        else:
            bar_color = DIM
            bar_chars = "—" * filled + "·" * (gauge_width - filled)

        sig_int = int(signal)
        noise_int = int(noise_floor)
        return f"{DIM}[{RESET}{bar_color}{bar_chars}{RESET}{DIM} {sig_int:3d}/{noise_int:2d}]{RESET}"

    def update_live_status(
        self,
        signal: float,
        noise_floor: float,
        is_speaking: bool,
        text: str = "",
        extra_status: str = "",
    ):
        """Render live compact status line with gauge, emoticon, and detected speech."""
        gauge = self.make_gauge(signal, noise_floor, is_speaking)
        icon = self.STATUS_ICONS["VOICE_DETECTED"] if is_speaking else self.STATUS_ICONS["LISTENING"]
        
        txt_display = ""
        if text.strip():
            txt_display = f" {BOLD}{WHITE}\"{text.strip()}\"{RESET}"
        elif extra_status:
            txt_display = f" {extra_status}"
        elif is_speaking:
            txt_display = f" {GREEN}(speaking...){RESET}"

        line = f"\r\033[K{gauge} {icon} {txt_display}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def update_sleep_status(
        self,
        curiosity_done: int = 0,
        curiosity_total: int = 0,
        notes_added: int = 0,
        is_online: bool = False,
        seconds_left: Optional[float] = None,
        dream: Optional[str] = None,
    ):
        """Render sleeping single line: 😴 Sleeping · curiosity 3/10 · notes +5 · dreaming: "..." · offline · 12:40 left"""
        conn_str = "online" if is_online else "offline"
        if seconds_left is not None and seconds_left >= 0:
            m = int(seconds_left // 60)
            s = int(seconds_left % 60)
            time_str = f" · {m:02d}:{s:02d} left"
        else:
            time_str = ""

        dream_str = ""
        if dream:
            clean_dream = " ".join(dream.split()).strip()
            # If dream is an inquiry or discovery text, clean leading prefix
            if clean_dream.startswith("Discovery regarding '"):
                clean_dream = clean_dream[21:].split("'")[0]
            dream_str = f" · {MAGENTA}dreaming: \"{clean_dream}\"{RESET}"

        line = f"\r\033[K😴 {BOLD}{BLUE}Sleeping{RESET} · curiosity {curiosity_done}/{curiosity_total} · notes +{notes_added}{dream_str} · {conn_str}{time_str}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def set_state(self, state: str, detail: str = ""):
        """Update status emoticon when state changes (THINKING, RESPONDING, ERROR, SLEEPING, MUTED, PTT)."""
        self.current_state = state.upper()
        icon = self.STATUS_ICONS.get(self.current_state, "●")
        
        if self.current_state == "THINKING":
            status_desc = f"{YELLOW}Thinking...{RESET}"
        elif self.current_state in ("RESPONDING", "SPEAKING"):
            icon = self.STATUS_ICONS["RESPONDING"]
            status_desc = f"{CYAN}Responding...{RESET}"
        elif self.current_state == "ERROR":
            icon = self.STATUS_ICONS["ERROR"]
            status_desc = f"{RED}Error: {detail}{RESET}"
        elif self.current_state == "SLEEPING":
            icon = self.STATUS_ICONS["SLEEPING"]
            status_desc = f"{BLUE}Sleeping...{RESET}"
        elif self.current_state == "MUTED":
            icon = self.STATUS_ICONS["MUTED"]
            status_desc = f"{YELLOW}Muted (press [M] to unmute){RESET}"
        elif self.current_state == "PTT":
            icon = self.STATUS_ICONS["PTT"]
            status_desc = f"{GREEN}Push-to-Talk (listening... press [Space] to finish){RESET}"
        else:
            status_desc = detail

        line = f"\r\033[K[        --/--] {icon} {status_desc}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def show_committed_user(self, text: str):
        """Finalize user utterance on a permanent clean line."""
        icon = self.STATUS_ICONS["VOICE_DETECTED"]
        sys.stdout.write(f"\r\033[K{icon} {BOLD}{WHITE}User:{RESET} {text.strip()}\n")
        sys.stdout.flush()

    def show_hal_reply(self, text: str):
        """Display HAL response cleanly on the line below."""
        icon = self.STATUS_ICONS["RESPONDING"]
        sys.stdout.write(f"{icon} {BOLD}{RED}HAL 9000:{RESET} {text.strip()}\n")
        sys.stdout.flush()

    def show_progress(self, step: int, total_steps: int, description: str = ""):
        """Render a single-line loading progress bar during initialization without text."""
        bar_len = 24
        ratio = max(0.0, min(1.0, step / float(total_steps)))
        filled = int(ratio * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = int(ratio * 100)

        # Render strictly on a single in-place line
        bar_line = f"\r\033[K{BOLD}{CYAN}Initializing HAL 9000 [{bar}] {pct:3d}%{RESET} {DIM}({step}/{total_steps}){RESET}"
        sys.stdout.write(bar_line)
        sys.stdout.flush()
        self._progress_active = True

        if step >= total_steps:
            self._progress_active = False
            sys.stdout.write(f"\r\033[K{BOLD}{GREEN}✔ HAL 9000 ready.{RESET}\n\n")
            sys.stdout.flush()

    def show_warning(self, message: str):
        """Prints warning line nicely without breaking layout."""
        icon = self.STATUS_ICONS.get("ERROR", "⚠️ ")
        if self._progress_active:
            sys.stdout.write(f"\r\033[K{icon} {BOLD}{YELLOW}Warning:{RESET} {message}\n")
        else:
            sys.stdout.write(f"\r\033[K{icon} {BOLD}{YELLOW}Warning:{RESET} {message}\n")
        sys.stdout.flush()

    def show_error(self, message: str):
        """Prints error line."""
        icon = self.STATUS_ICONS["ERROR"]
        if self._progress_active:
            sys.stdout.write(f"\r\033[K{icon} {BOLD}{RED}Error:{RESET} {message}\n")
        else:
            sys.stdout.write(f"\r\033[K{icon} {BOLD}{RED}Error:{RESET} {message}\n")
        sys.stdout.flush()

    def show_info(self, message: str):
        """Prints diagnostic info."""
        if not self._progress_active:
            sys.stdout.write(f"\r\033[K{DIM}[INFO] {message}{RESET}\n")
            sys.stdout.flush()

    def restore(self):
        """Restores terminal cursor and state."""
        sys.stdout.write(f"\r\033[K{RESET}\n")
        sys.stdout.flush()


