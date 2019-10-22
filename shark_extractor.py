import cv2
import numpy as np

def create_first_half_of_edge_mask(first_edge_mask, kernel = np.ones((5,5),np.uint8), max_thresh = 228, min_thresh = 198):
    first_edge_mask = cv2.cvtColor(first_edge_mask, cv2.COLOR_BGR2GRAY)
    first_edge_mask = cv2.equalizeHist(first_edge_mask)
    first_edge_mask = cv2.bilateralFilter(first_edge_mask,9,75,75)

    # Threshold image to be between max and min
    first_edge_mask[first_edge_mask > max_thresh] = 0
    first_edge_mask[first_edge_mask < min_thresh] = 0
    first_edge_mask[first_edge_mask !=0] = 255

    # Expand Thresholded image
    first_edge_mask = cv2.dilate(first_edge_mask, kernel, iterations = 2)
    return first_edge_mask

def create_second_half_of_edge_mask(second_edge_mask, max_thresh = 120, min_thresh = 90):
    # Spread and Blur values
    second_edge_mask = cv2.equalizeHist(second_edge_mask)
    second_edge_mask = cv2.bilateralFilter(second_edge_mask,9,75,75)

    # Threshold image to be between 90 and 120
    second_edge_mask[second_edge_mask > max_thresh] = 0
    second_edge_mask[second_edge_mask < min_thresh] = 0
    second_edge_mask[second_edge_mask != 0] = 255
    return second_edge_mask

def combine_edge_masks(edge_mask1, edge_mask2, kernel = np.ones((5,5),np.uint8)):
    # Combine and blur the two masks to get shark outline and remove unwanted objects
    combined_edge_mask = cv2.bitwise_and(edge_mask1, edge_mask2)
    full_edge_mask = cv2.dilate(combined_edge_mask,kernel, iterations = 2)
    full_edge_mask = cv2.bilateralFilter(full_edge_mask,9,75,75)
    return full_edge_mask

# Fill in the edges so that they can be used as a mask
def fill_mask_edges(full_edge_mask, kernel = np.ones((5,5),np.uint8)):
    # Fill edges
    h, w = full_edge_mask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    filled_mask = full_edge_mask.copy()
    cv2.floodFill(filled_mask, mask, (0,0), 255)
    filled_mask = cv2.bitwise_not(filled_mask)

    full_mask = cv2.bitwise_or(filled_mask, full_edge_mask)

    # Erode and Dilate to remove unwanted objects
    full_mask_erosion = cv2.erode(full_mask,kernel,iterations = 6)
    full_mask_erosion = cv2.bilateralFilter(full_mask_erosion,9,75,75)
    full_mask_erosion = cv2.dilate(full_mask_erosion,kernel,iterations = 4)
    full_mask_erosion = cv2.bilateralFilter(full_mask_erosion,9,75,75)
    return full_mask_erosion

def create_detailed_image(first_channel,second_channel,third_channel):
    return cv2.merge([first_channel,second_channel,third_channel])

# This gets the lowest and highest value of both x and y
# Based on whether the pixel is pure black or not
# Then returns a sub image of that range

def crop_image(image, mask):
    y_index, x_index = np.where(mask != 0)
    # Both arrays are sorted based on the contents of y_index
    # So x_index needs to be resorted based on its own contents
    x_index.sort()
    first_x = x_index[0]
    first_y = y_index[0]
    last_x = x_index[len(x_index)-1]
    last_y = y_index[len(y_index)-1]
    return image[first_y:last_y, first_x:last_x]

# This just replaces the black from the mask with white
def whiteout_background(image):
    image[image == 0] = 255
    return image


def extract_shark(file_path = "./images/Shark 1.PNG"):
    # Read in image
    image = cv2.imread(file_path, 1) 

    # Create necessary color spaces and images from src image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    xyz = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    

    luv_first, _, luv_third = cv2.split(luv)
    _,         _, lab_third = cv2.split(lab)

    luv_third_eq = cv2.equalizeHist(luv_third)

    _, _, r = cv2.split(image)
    

    _, _, xyz_third = cv2.split(xyz)
    detailed_image = create_detailed_image(cv2.equalizeHist(luv_first),cv2.equalizeHist(luv_first),r)
    detailed_image_gray = create_detailed_image(cv2.equalizeHist(luv_first),cv2.equalizeHist(luv_first),cv2.equalizeHist(luv_first))

    # Create first half of edge mask
    first_edge_mask = create_first_half_of_edge_mask(cv2.merge([r, luv_third_eq, lab_third]))

    # Create second half of edge mask
    second_edge_mask = create_second_half_of_edge_mask(xyz_third)

    # Combine Edge Masks
    full_edge_mask = combine_edge_masks(first_edge_mask, second_edge_mask)

    # Fill mask edges
    full_mask = fill_mask_edges(full_edge_mask)

    # Mask Detailed Image
    masked = cv2.bitwise_or(detailed_image, detailed_image, mask=full_mask)

    # Crop image to size of shark
    cropped_image = crop_image(masked, full_mask)

    # Change image background from black to white
    white_background_img = whiteout_background(cropped_image)
    cv2.imshow("Final Cropped Image with White background", white_background_img)

    masked_gray = cv2.bitwise_or(detailed_image_gray, detailed_image_gray, mask=full_mask)

    # Crop image to size of shark
    cropped_image_gray = crop_image(masked_gray, full_mask)

    # Change image background from black to white
    white_background_img_gray = whiteout_background(cropped_image_gray)
    cv2.imshow("Final Cropped Image with White background Gray", white_background_img_gray)
    return cropped_image, cropped_image_gray

def main():
    # Extract sharks
    first_shark_color, first_shark_gray = extract_shark("./images/shark1.png")
    second_shark_color, second_shark_gray = extract_shark("./images/shark2.png")
    cv2.imshow("Shark 1 extracted Color", first_shark_color)
    cv2.imshow("Shark 2 extracted Color", second_shark_color)
    cv2.imshow("Shark 1 extracted Gray", first_shark_gray)
    cv2.imshow("Shark 2 extracted Gray", second_shark_gray)

    # Keep image on screen
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()