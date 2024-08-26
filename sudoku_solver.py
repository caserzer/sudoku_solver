import cv2
import numpy as np
import pyautogui
import os
import pytesseract
import numpy as np
import argparse
import copy


def find_board( image , x , y , width, offsetx = 0, offsety = 0):
    # 从image 中提取以x,y为左上角，宽度为9倍的width,高度为9倍的width的区域,并保存为board.png
    board = image[y-offsety*w:y-offsety*w+9*(width+4), x-offsetx*w:x-offsetx*w+9*(width+4)]
    cv2.imwrite(working_dir + '/board.png', board)

    # 将board 按照9x9的格子切分，并保存到working目录下，每个格子的文件名为cell_{行号}_{列号}.png
    rows, cols, _ = board.shape
    cell_width = cols // 9
    cell_height = rows // 9

    # init a 9x9 board
    board_nums = [[0] * 9 for _ in range(9)]
    for i in range(9):
        for j in range(9):
            cell = board[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            border_percent = 0.15
            border_height = int(cell.shape[0] * border_percent)
            border_width = int(cell.shape[1] * border_percent)
            cell = cell[border_height:-border_height, border_width:-border_width]
            # cell 二值化
            threshold_value = 127  # 阈值，可以根据需要调整
            max_value = 255  # 最大值，通常为255
            _, cell = cv2.threshold(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY), threshold_value, max_value, cv2.THRESH_BINARY)

            #cv2.imwrite(working_dir + f'/cell_{i}_{j}.png', cell)
            board_nums[i][j] = recognize_digital(cell)
            #cv2.imwrite(working_dir + f'/cell_{i}_{j}.png', cell)
    return board_nums

def recognize_digital(image,show_log = False):
    flattened_image = image.reshape(-1, image.shape[-1])
    first_pixel = flattened_image[0]
    if np.all(flattened_image == first_pixel):
        return 0

    custom_config = r'--psm 10 --oem 0 -c tessedit_char_whitelist=123456789'
    text = pytesseract.image_to_string(image, config=custom_config).strip()
    if text.isdigit():
        return int(text)
    else:
        return 0

#solve the sudoku
def solve_sudoku(board):
    # Find the next empty cell
    row, col = find_empty_cell(board)

    # If there are no empty cells, the puzzle is solved
    if row is None:
        return True

    # Try each number from 1 to 9
    for num in range(1, 10):
        # Check if the number is valid in the current position
        if is_valid(board, row, col, num):
            # Place the number in the current position
            board[row][col] = num

            # Recursively solve the rest of the puzzle
            if solve_sudoku(board):
                return True

            # If the current number doesn't lead to a solution, backtrack
            board[row][col] = 0

    # If no number from 1 to 9 is valid, the puzzle is unsolvable
    return False

# Function to find the next empty cell in the board
def find_empty_cell(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col
    return None, None

# Function to check if a number is valid in a given position
def is_valid(board, row, col, num):
    # Check if the number is already present in the same row
    if num in board[row]:
        return False

    # Check if the number is already present in the same column
    if num in [board[i][col] for i in range(9)]:
        return False

    # Check if the number is already present in the same 3x3 box
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    if num in [board[i][j] for i in range(box_row, box_row + 3) for j in range(box_col, box_col + 3)]:
        return False

    return True

def fill_board(board, original_board, left, top, width):
    pyautogui.moveTo((left)//2 , (top)//2 )
    pyautogui.leftClick()

    for i in range(9):
        for j in range(9):
            if original_board[i][j] != 0:
                continue
            number = board[i][j]
            pyautogui.moveTo((left + width*j+ width// 2)//2 +5 , (top + width*i+ width// 2)//2 +5 )
            pyautogui.leftClick()
            # pyautogui.typewrite(str(number), interval=0.1)
            pyautogui.press(str(number))

# 创建解析器对象
parser = argparse.ArgumentParser(description="Sudoku Solver")

# 添加命令行参数
parser.add_argument('--offsetx', type=int, required=True, help='输入列偏移量', default=0)
parser.add_argument('--offsety', type=int, required=True, help='输入行偏移量', default=0)
args = parser.parse_args()

print(f'offsetx: {args.offsetx}, offsety: {args.offsety}')

current_dir = os.getcwd()

working_dir = current_dir + "/working"

found_board = False

pyautogui.PAUSE = 0.01
# Create a 2D list to represent the sudoku board
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]


screenshot_file  = working_dir + "/screenshot.png"
print("Taking screenshot...")
image = pyautogui.screenshot()

image.save(screenshot_file)

# 读取图片
image = cv2.imread(screenshot_file)
print('image shape:', image.shape)
# 将 e9ca43 转换为 BGR 格式
target_color_bgr_lower = np.array([0, 193, 220])
target_color_bgr_upper = np.array([50, 213, 239])

# 创建一个掩码，寻找与目标颜色相匹配的区域
mask = cv2.inRange(image, target_color_bgr_lower, target_color_bgr_upper)

# 查找掩码中的轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

Left = 0
Top = 0
Width = 0

# 如果找到了轮廓
if contours:
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w == h :
            h = h -1
        else:
            h = min(w,h)
        # w=w -1
        # h = h - 1
        print(f"Found target color at: X={x}, Y={y}, Width={w}, Height={h}")
        # 在图像上绘制矩形框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        board = find_board(image, x, y, h, args.offsetx, args.offsety)
        found_board = True
        Left = x - args.offsetx*w
        Top = y - args.offsety*w
        Width = w


if found_board:
    print("Found Sudoku Board")
    for row in board:
        print(row)

    print("Sudoku Board Result:")
    origianl_board = copy.deepcopy(board)
    solve_sudoku(board)

    for i in range(9):
        for j in range(9):
            print(board[i][j], end=' ')

    fill_board(board, origianl_board, Left, Top, Width)    
