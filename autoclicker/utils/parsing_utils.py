# utils/parsing_utils.py
import logging

logger = logging.getLogger(__name__)

def parse_tuple_str(tuple_str: str | None, expected_len: int, item_type=int) -> tuple | None:
    """
    Parses a string like "8,8" or "50,150" into a tuple of specified type and length.
    Returns None on error, empty string, or None input.
    """
    if not isinstance(tuple_str, str) or not tuple_str.strip():
        return None
    parts = tuple_str.strip().split(',')
    if len(parts) != expected_len:
        logger.warning(f"Parsing tuple string '{tuple_str}': Expected {expected_len} parts, got {len(parts)}.")
        return None
    try:
        parsed_items = [item_type(p.strip()) for p in parts]
        return tuple(parsed_items)
    except (ValueError, TypeError):
        logger.warning(f"Parsing tuple string '{tuple_str}': Error converting parts to {item_type.__name__}.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing tuple string '{tuple_str}': {e}", exc_info=True)
        return None
    
# Auto Clicker Enhanced
# Copyright (C) <2025> <Đinh Khởi Minh>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.