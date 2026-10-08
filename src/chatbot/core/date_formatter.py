"""Date formatting utilities."""

import re


def format_date(date_str: str) -> str:
    """
    Format date string to YYYYMMDD format.

    Replicates the JavaScript logic from Tambora.yml:
    - Removes whitespace and newlines
    - Extracts only digits
    - Returns first 8 digits

    Args:
        date_str: Date string in any format

    Returns:
        Date in YYYYMMDD format (8 digits)
    """
    if not date_str:
        return ""

    # Remove whitespace and newlines
    cleaned = date_str.replace("\r", "").replace("\n", "").strip()

    # Extract only digits
    digits_only = re.sub(r"\D", "", cleaned)

    # Return first 8 digits
    return digits_only[:8]
