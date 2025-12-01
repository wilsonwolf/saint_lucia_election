#!/usr/bin/env python3
"""
Election Night Alert Utilities

Provides notification functionality for election night monitoring:
- macOS desktop notifications
- Sound alerts for critical events
- State change detection to avoid repeated alerts
"""

import subprocess
from typing import Dict, List, Optional, Tuple


def send_notification(
    title: str,
    message: str,
    sound: bool = False,
    sound_name: str = "Ping"
) -> bool:
    """
    Send a macOS desktop notification.

    Args:
        title: Notification title
        message: Notification body text
        sound: Whether to play a sound
        sound_name: Name of system sound to play (e.g., "Ping", "Glass", "Hero")

    Returns:
        True if notification was sent successfully
    """
    try:
        # Build AppleScript command for notification
        script = f'display notification "{message}" with title "{title}"'
        if sound:
            script += f' sound name "{sound_name}"'

        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5
        )
        return True
    except Exception as e:
        print(f"[ALERT ERROR] Failed to send notification: {e}")
        return False


def play_alert_sound(sound_type: str = "critical") -> bool:
    """
    Play an alert sound.

    Args:
        sound_type: "critical" for urgent alerts, "warning" for less urgent

    Returns:
        True if sound played successfully
    """
    # Map sound types to system sounds
    sounds = {
        "critical": "/System/Library/Sounds/Sosumi.aiff",  # Attention-grabbing
        "warning": "/System/Library/Sounds/Ping.aiff",     # Subtle ping
        "success": "/System/Library/Sounds/Glass.aiff",    # Pleasant chime
    }

    sound_path = sounds.get(sound_type, sounds["warning"])

    try:
        subprocess.run(
            ["afplay", sound_path],
            capture_output=True,
            timeout=5
        )
        return True
    except Exception as e:
        print(f"[ALERT ERROR] Failed to play sound: {e}")
        return False


def check_and_alert(
    current_analysis: Dict[str, Dict],
    previous_analysis: Optional[Dict[str, Dict]] = None,
    enable_sound: bool = True,
    enable_notification: bool = True
) -> List[Tuple[str, str, str]]:
    """
    Check for state changes and send appropriate alerts.

    Only alerts on STATE CHANGES to avoid repeated notifications:
    - CRITICAL: Constituency just became flip_projected
    - WARNING: Constituency just became at_risk

    Args:
        current_analysis: Current constituency analysis dict
        previous_analysis: Previous analysis dict (None if first run)
        enable_sound: Whether to play sound alerts
        enable_notification: Whether to send desktop notifications

    Returns:
        List of (constituency_name, alert_type, message) tuples
    """
    alerts_sent = []

    if previous_analysis is None:
        previous_analysis = {}

    for const_name, analysis in current_analysis.items():
        current_status = analysis.get("status", "unknown")
        previous_status = previous_analysis.get(const_name, {}).get("status", "unknown")

        # Skip if status hasn't changed
        if current_status == previous_status:
            continue

        baseline_winner = analysis.get("baseline_winner", "?")
        swing = analysis.get("swing_analysis", {}).get("avg_swing_all_districts", 0)
        breakeven = analysis.get("breakeven_swing_pct", 0)

        # CRITICAL: Just became flip_projected
        if current_status == "flip_projected" and previous_status != "flip_projected":
            alert_type = "CRITICAL"
            message = f"{const_name}: FLIP PROJECTED - {baseline_winner} losing (swing {swing:+.1f}% vs breakeven {-breakeven:.1f}%)"

            if enable_notification:
                send_notification(
                    f"FLIP PROJECTED: {const_name}",
                    f"{baseline_winner} seat projected to flip! Swing: {swing:+.1f}%",
                    sound=False  # We'll play our own sound
                )

            if enable_sound:
                play_alert_sound("critical")

            alerts_sent.append((const_name, alert_type, message))

        # WARNING: Just became at_risk
        elif current_status == "at_risk" and previous_status not in ["at_risk", "flip_projected"]:
            alert_type = "WARNING"
            message = f"{const_name}: AT RISK - swing {swing:+.1f}% approaching breakeven {-breakeven:.1f}%"

            if enable_notification:
                send_notification(
                    f"AT RISK: {const_name}",
                    f"{baseline_winner} seat at risk. Swing: {swing:+.1f}%",
                    sound=True,
                    sound_name="Ping"
                )

            alerts_sent.append((const_name, alert_type, message))

        # INFO: Recovered from at_risk/flip_projected back to holding
        elif current_status == "holding" and previous_status in ["at_risk", "flip_projected"]:
            alert_type = "INFO"
            message = f"{const_name}: RECOVERED - now holding (swing {swing:+.1f}%)"

            if enable_notification:
                send_notification(
                    f"RECOVERED: {const_name}",
                    f"{baseline_winner} seat now safe. Swing: {swing:+.1f}%",
                    sound=True,
                    sound_name="Glass"
                )

            alerts_sent.append((const_name, alert_type, message))

    return alerts_sent


def format_alert_summary(alerts: List[Tuple[str, str, str]]) -> str:
    """
    Format a list of alerts for console output.

    Args:
        alerts: List of (constituency_name, alert_type, message) tuples

    Returns:
        Formatted string for console display
    """
    if not alerts:
        return ""

    lines = ["", "ALERTS:"]

    # Sort by severity (CRITICAL first, then WARNING, then INFO)
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    sorted_alerts = sorted(alerts, key=lambda x: severity_order.get(x[1], 99))

    for const_name, alert_type, message in sorted_alerts:
        lines.append(f"  [{alert_type}] {message}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test notifications
    print("Testing notifications...")

    send_notification(
        "Election Night Monitor",
        "Test notification - alerts are working!",
        sound=True
    )

    print("Test complete!")
