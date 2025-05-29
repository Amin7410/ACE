# utils/image_processing.py
import logging
import cv2 
import numpy as np
from .parsing_utils import parse_tuple_str 

logger = logging.getLogger(__name__)

DEFAULT_GAUSSIAN_KERNEL = (3, 3)
DEFAULT_MEDIAN_KERNEL = 3
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_SIZE = (8, 8)
DEFAULT_BILATERAL_D = 9
DEFAULT_BILATERAL_SIGMA = 75
DEFAULT_CANNY_T1 = 50
DEFAULT_CANNY_T2 = 150

def preprocess_for_image_matching(image_np: np.ndarray | None, pp_params: dict) -> np.ndarray | None:
    """
    Áp dụng một quy trình tiền xử lý có thể cấu hình, phù hợp cho việc khớp ảnh
    (Template Matching, Feature Matching).

    Args:
        image_np: Ảnh đầu vào dưới dạng NumPy array (BGR, BGRA, hoặc Grayscale). Có thể là None.
        pp_params: Một dictionary chứa các tham số tiền xử lý như:
            'grayscale': bool (mặc định: True)
            'binarization': bool (mặc định: False - dùng Otsu)
            'gaussian_blur': bool (mặc định: False)
            'gaussian_blur_kernel': str (ví dụ: "3,3")
            'median_blur': bool (mặc định: False)
            'median_blur_kernel': int (ví dụ: 3)
            'clahe': bool (mặc định: False)
            'clahe_clip_limit': float (ví dụ: 2.0)
            'clahe_tile_grid_size': str (ví dụ: "8,8")
            'bilateral_filter': bool (mặc định: False)
            'bilateral_d': int (ví dụ: 9)
            'bilateral_sigma_color': float (ví dụ: 75)
            'bilateral_sigma_space': float (ví dụ: 75)
            'canny_edges': bool (mặc định: False)
            'canny_threshold1': float (ví dụ: 50)
            'canny_threshold2': float (ví dụ: 150)

    Returns:
        Ảnh đã tiền xử lý dưới dạng NumPy array, hoặc ảnh gốc nếu không có
        xử lý nào được áp dụng hoặc khả thi, hoặc None nếu đầu vào không hợp lệ.
    """
    if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
        logger.warning("preprocess_for_image_matching nhận ảnh đầu vào không hợp lệ.")
        return None

    try:
        processed = image_np.copy()
        logger.debug(f"Image Matching Preprocessing Start. Input shape: {processed.shape}")

        use_grayscale = pp_params.get('grayscale', True) 
        is_gray = False
        if use_grayscale:
            if processed.ndim == 3:
                try:
                    code = cv2.COLOR_BGRA2GRAY if processed.shape[2] == 4 else cv2.COLOR_BGR2GRAY
                    processed = cv2.cvtColor(processed, code)
                    is_gray = True
                    logger.debug("Applied Grayscale")
                except cv2.error as e:
                    logger.warning(f"Lỗi chuyển đổi Grayscale: {e}. Tiếp tục với ảnh gốc nếu có thể.")
            elif processed.ndim == 2:
                is_gray = True 
                logger.debug("Input image is already grayscale.")

        can_do_2d_filtering = is_gray
        if not can_do_2d_filtering and (pp_params.get('gaussian_blur', False) or
                                      pp_params.get('median_blur', False) or
                                      pp_params.get('clahe', False) or
                                      pp_params.get('canny_edges', False) or
                                      pp_params.get('binarization', False)):
            logger.warning("Không thể áp dụng các bộ lọc/xử lý 2D vì ảnh không phải là ảnh xám.")

        try:
            if can_do_2d_filtering and pp_params.get('gaussian_blur', False):
                kernel_str = pp_params.get('gaussian_blur_kernel', "3,3")
                kernel = parse_tuple_str(kernel_str, 2, int) or DEFAULT_GAUSSIAN_KERNEL
                if all(k > 0 and k % 2 == 1 for k in kernel):
                    processed = cv2.GaussianBlur(processed, kernel, 0)
                    logger.debug(f"Applied Gaussian Blur (Kernel: {kernel})")
                else:
                    logger.warning(f"Kernel Gaussian không hợp lệ '{kernel_str}', bỏ qua bước làm mờ.")

            if can_do_2d_filtering and pp_params.get('median_blur', False):
                kernel_size = pp_params.get('median_blur_kernel', DEFAULT_MEDIAN_KERNEL)
                if kernel_size > 0 and kernel_size % 2 == 1:
                    if processed.shape[0] > kernel_size and processed.shape[1] > kernel_size:
                        processed = cv2.medianBlur(processed, kernel_size)
                        logger.debug(f"Applied Median Blur (Kernel: {kernel_size})")
                    else:
                        logger.warning(f"Kích thước ảnh nhỏ hơn kernel Median ({kernel_size}), bỏ qua bước làm mờ.")
                else:
                    logger.warning(f"Kích thước kernel Median không hợp lệ {kernel_size}, bỏ qua bước làm mờ.")

            if pp_params.get('bilateral_filter', False):
                d = pp_params.get('bilateral_d', DEFAULT_BILATERAL_D)
                sc = pp_params.get('bilateral_sigma_color', DEFAULT_BILATERAL_SIGMA)
                ss = pp_params.get('bilateral_sigma_space', DEFAULT_BILATERAL_SIGMA)
                processed = cv2.bilateralFilter(processed, d, sc, ss)
                logger.debug(f"Applied Bilateral Filter (d={d}, sc={sc}, ss={ss})")

        except cv2.error as e: logger.warning(f"Bước Làm mờ/Lọc thất bại: {e}")
        except Exception as e: logger.error(f"Lỗi không mong muốn trong quá trình làm mờ: {e}", exc_info=True)

        try:
            if can_do_2d_filtering and pp_params.get('clahe', False):
                clip = pp_params.get('clahe_clip_limit', DEFAULT_CLAHE_CLIP_LIMIT)
                tile_str = pp_params.get('clahe_tile_grid_size', "8,8")
                tile = parse_tuple_str(tile_str, 2, int) or DEFAULT_CLAHE_TILE_SIZE
                if all(s > 0 for s in tile):
                    clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
                    processed = clahe_obj.apply(processed)
                    logger.debug(f"Applied CLAHE (Clip: {clip}, Tile: {tile})")
                else:
                    logger.warning(f"Kích thước tile CLAHE không hợp lệ '{tile_str}', bỏ qua CLAHE.")
        except cv2.error as e: logger.warning(f"CLAHE thất bại: {e}")
        except Exception as e: logger.error(f"Lỗi không mong muốn trong quá trình CLAHE: {e}", exc_info=True)

        use_canny = False 
        try:
            if can_do_2d_filtering and pp_params.get('canny_edges', False):
                t1 = pp_params.get('canny_threshold1', DEFAULT_CANNY_T1)
                t2 = pp_params.get('canny_threshold2', DEFAULT_CANNY_T2)
                processed = cv2.Canny(processed, t1, t2)
                use_canny = True 
                logger.debug(f"Applied Canny Edges (T1: {t1}, T2: {t2})")
        except cv2.error as e: logger.warning(f"Phát hiện cạnh Canny thất bại: {e}")
        except Exception as e: logger.error(f"Lỗi không mong muốn trong quá trình Canny: {e}", exc_info=True)

        try:
            if can_do_2d_filtering and pp_params.get('binarization', False) and not use_canny:
                _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                logger.debug("Applied Otsu Binarization")
        except cv2.error as e: logger.warning(f"Nhị phân hóa thất bại: {e}")
        except Exception as e: logger.error(f"Lỗi không mong muốn trong quá trình Nhị phân hóa: {e}", exc_info=True)


        logger.debug(f"Image Matching Preprocessing End. Output shape: {processed.shape}")
        return processed

    except Exception as main_ex:
        logger.error(f"Lỗi không mong muốn trong preprocess_for_image_matching: {main_ex}", exc_info=True)
        return image_np 

def preprocess_for_ocr(image_np: np.ndarray | None, pp_params: dict) -> np.ndarray | None:
    """
    Áp dụng một quy trình tiền xử lý có thể cấu hình, được tối ưu hóa cho OCR.

    Args:
        image_np: Ảnh đầu vào dưới dạng NumPy array (BGR, BGRA, hoặc Grayscale). Có thể là None.
        pp_params: Một dictionary chứa các tham số tiền xử lý như:
            'grayscale': bool (mặc định: True)
            'adaptive_threshold': bool (mặc định: True) # Ưu tiên ngưỡng thích ứng
            'binarization': bool (mặc định: False) # Chỉ dùng nếu adaptive_threshold là False
            'ocr_upscale_factor': float (mặc định: 1.0)
            'median_blur': bool (mặc định: True)
            'median_blur_kernel': int (ví dụ: 3)
            'gaussian_blur': bool (mặc định: False)
            'gaussian_blur_kernel': str (ví dụ: "3,3")
            'clahe': bool (mặc định: False)
            'clahe_clip_limit': float (ví dụ: 2.0)
            'clahe_tile_grid_size': str (ví dụ: "8,8")
            'bilateral_filter': bool (mặc định: False)
            # ... các tham số bilateral khác nếu dùng cho OCR ...

    Returns:
        Ảnh đã tiền xử lý dưới dạng NumPy array, hoặc ảnh gốc nếu không có
        xử lý nào được áp dụng hoặc khả thi, hoặc None nếu đầu vào không hợp lệ.
    """
    if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
        logger.warning("preprocess_for_ocr nhận ảnh đầu vào không hợp lệ.")
        return None

    try:
        processed = image_np.copy()
        logger.debug(f"OCR Preprocessing Start. Input shape: {processed.shape}")

        try:
            upscale_factor = pp_params.get('ocr_upscale_factor', 1.0)
            try:
                upscale_factor = float(upscale_factor)
            except (ValueError, TypeError):
                logger.warning(f"ocr_upscale_factor không hợp lệ '{pp_params.get('ocr_upscale_factor')}', sử dụng 1.0.")
                upscale_factor = 1.0

            if upscale_factor > 1.0:
                    new_width = int(processed.shape[1] * upscale_factor)
                    new_height = int(processed.shape[0] * upscale_factor)
                    processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    logger.debug(f"Applied Upscaling (Factor: {upscale_factor:.2f}), New shape: {processed.shape}")
        except cv2.error as e: logger.warning(f"Upscaling thất bại: {e}")
        except Exception as e: logger.error(f"Lỗi không mong muốn trong quá trình Upscaling: {e}", exc_info=True)

        use_grayscale = pp_params.get('grayscale', True) 
        is_gray = False
        if use_grayscale or processed.ndim != 2:
            if processed.ndim == 3:
                try:
                    code = cv2.COLOR_BGRA2GRAY if processed.shape[2] == 4 else cv2.COLOR_BGR2GRAY
                    processed = cv2.cvtColor(processed, code)
                    is_gray = True
                    logger.debug("Applied Grayscale for OCR")
                except cv2.error as e:
                    logger.warning(f"Lỗi chuyển đổi Grayscale cho OCR: {e}. Tiếp tục với ảnh gốc nếu có thể.")
            elif processed.ndim == 2:
                 is_gray = True
                 logger.debug("Input image is already grayscale for OCR.")
            else:
                logger.warning(f"Không thể áp dụng tiền xử lý OCR, định dạng ảnh không phù hợp (ndim={processed.ndim}).")
                return processed 
        if not is_gray:
             logger.warning("Ảnh không phải Grayscale sau khi cố gắng chuyển đổi, bỏ qua các bước tiền xử lý OCR tiếp theo.")
             return processed

        try:
            if pp_params.get('gaussian_blur', False): 
                kernel_str = pp_params.get('gaussian_blur_kernel', "3,3")
                kernel = parse_tuple_str(kernel_str, 2, int) or DEFAULT_GAUSSIAN_KERNEL
                if all(k > 0 and k % 2 == 1 for k in kernel):
                    processed = cv2.GaussianBlur(processed, kernel, 0)
                    logger.debug(f"Applied Gaussian Blur (Kernel: {kernel}) for OCR")
                else: logger.warning(f"Kernel Gaussian không hợp lệ '{kernel_str}', bỏ qua blur cho OCR.")

            if pp_params.get('median_blur', True): 
                kernel_size = pp_params.get('median_blur_kernel', DEFAULT_MEDIAN_KERNEL)
                if kernel_size > 0 and kernel_size % 2 == 1:
                    if processed.shape[0] > kernel_size and processed.shape[1] > kernel_size:
                        processed = cv2.medianBlur(processed, kernel_size)
                        logger.debug(f"Applied Median Blur (Kernel: {kernel_size}) for OCR")
                    else: logger.warning(f"Kích thước ảnh nhỏ hơn kernel Median ({kernel_size}), bỏ qua blur cho OCR.")
                else: logger.warning(f"Kích thước kernel Median không hợp lệ {kernel_size}, bỏ qua blur cho OCR.")

            if pp_params.get('bilateral_filter', False):
                d = pp_params.get('bilateral_d', 5) 
                sc = pp_params.get('bilateral_sigma_color', DEFAULT_BILATERAL_SIGMA)
                ss = pp_params.get('bilateral_sigma_space', DEFAULT_BILATERAL_SIGMA)
                processed = cv2.bilateralFilter(processed, d, sc, ss)
                logger.debug(f"Applied Bilateral Filter (d={d}, sc={sc}, ss={ss}) for OCR")

        except cv2.error as e: logger.warning(f"Bước làm mờ cho OCR thất bại: {e}")
        except Exception as e: logger.error(f"Lỗi không mong muốn trong quá trình làm mờ OCR: {e}", exc_info=True)

        try:
            if pp_params.get('clahe', False): 
                clip = pp_params.get('clahe_clip_limit', DEFAULT_CLAHE_CLIP_LIMIT)
                tile_str = pp_params.get('clahe_tile_grid_size', "8,8")
                tile = parse_tuple_str(tile_str, 2, int) or DEFAULT_CLAHE_TILE_SIZE
                if all(s > 0 for s in tile):
                    clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
                    processed = clahe_obj.apply(processed)
                    logger.debug(f"Applied CLAHE (Clip: {clip}, Tile: {tile}) for OCR")
                else: logger.warning(f"Kích thước tile CLAHE không hợp lệ '{tile_str}', bỏ qua CLAHE cho OCR.")
        except cv2.error as e: logger.warning(f"CLAHE cho OCR thất bại: {e}")
        except Exception as e: logger.error(f"Lỗi không mong muốn trong quá trình CLAHE OCR: {e}", exc_info=True)


        try:
            if pp_params.get('adaptive_threshold', True): 
                 block_size = 21 
                 C = 9 
                 processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, block_size, C)
                 logger.debug("Applied Adaptive Threshold (Gaussian) for OCR")
            elif pp_params.get('binarization', False): 
                 _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                 logger.debug("Applied Otsu Binarization for OCR")
        except cv2.error as e: logger.warning(f"Thresholding cho OCR thất bại: {e}")
        except Exception as e: logger.error(f"Lỗi không mong muốn trong quá trình Thresholding OCR: {e}", exc_info=True)

        logger.debug(f"OCR Preprocessing End. Output shape: {processed.shape}")
        return processed

    except Exception as main_ex:
        logger.error(f"Lỗi không mong muốn trong preprocess_for_ocr: {main_ex}", exc_info=True)
        return image_np 
    

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
