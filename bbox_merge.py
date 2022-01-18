import copy
import torch
import numpy as np
import scipy.optimize

np.set_printoptions(suppress=True, precision=2)


def bbox_merge(boxes: torch.Tensor,
        cat_main: int,
        cat_sub: int,
        cat_target: int,
        min_iou: float = 0.1) -> torch.Tensor:
    """
    Merge two bounding boxes (of selected categories) into a single bounding box,
    whose size is the outermost dimensions of the two merged bboxes.
    Uses Hungarian algorithm to match them.

    Arguments:
        boxes: torch.Tensor array of size (n, 6), where n is number of bounding box detections
        cat_main: category number of relatively LESS frequent category
        cat_sub: category number of relatively MORE frquent category
        cat_target: category number of newly created object; make sure it does not collide with pre-existing labels
        min_iou: Intersection-over-union threshold value for matching

    Returns:
        torch.Tensor array of size (n-m, 6), where m is number of successful merges
            Resulting bbox will have the outermost dimensions of the two joined bboxes.
    """
    dtype = boxes.dtype
    device = boxes.device
    
    boxes = boxes.detach().cpu().numpy()
    main_boxes = boxes[boxes[:,5]==cat_main]
    sub_boxes = boxes[boxes[:,5]==cat_sub]
    other_boxes = boxes[np.logical_and(boxes[:,5]!=cat_main, boxes[:,5]!=cat_sub)]
    
    # coordinates are pixel values in order: (xmin,ymin,xmax,ymax)
    main_coords = main_boxes[:,:4]
    sub_coords = sub_boxes[:,:4]
    
    intersections = iou(main_coords, sub_coords)
    
    assignment = scipy.optimize.linear_sum_assignment(intersections, maximize=True)
    
    main_boxes_const = copy.copy(main_boxes)
    sub_boxes_const = copy.copy(sub_boxes)
    
    merged_coords = np.empty((0,6))

    main_indexes_to_delete = []
    sub_indexes_to_delete = []

    for i in range(len(assignment[0])):
        main_index = assignment[0][i]
        sub_index = assignment[1][i]
        main_indexes_to_delete.append(main_index)
        sub_indexes_to_delete.append(sub_index)
        if intersections[main_index,sub_index] == 0: # IOU is zero; should not be assigned
            continue
        else:
            matched_pair = np.vstack((main_boxes_const[main_index], sub_boxes_const[sub_index]))
            xmin = np.amin(matched_pair, axis=0)[0]
            ymin = np.amin(matched_pair, axis=0)[1]
            xmax = np.amax(matched_pair, axis=0)[2]
            ymax = np.amax(matched_pair, axis=0)[3]
            avg_conf = np.mean(matched_pair, axis=0)[4]

            merged_coords = np.vstack((merged_coords, np.array([xmin,ymin,xmax,ymax,avg_conf,cat_target])))
            
    # delete the two sources of merged new object
    main_boxes = np.delete(main_boxes, main_indexes_to_delete, axis=0)
    sub_boxes = np.delete(sub_boxes, sub_indexes_to_delete, axis=0)
            
    return torch.tensor(np.vstack((main_boxes, sub_boxes, merged_coords, other_boxes)), dtype=dtype ,device=device)

def iou(boxes1, boxes2):
    '''
    Vectorized intersection-over-union
    '''
    x11,y11,x12,y12 = np.split(boxes1, 4, axis=1)
    x21,y21,x22,y22 = np.split(boxes2, 4, axis=1)
    
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum((xB - xA + 1), 0 ) * np.maximum((yB-yA + 1), 0)

    boxAArea = (x12-x11+1) * (y12-y11+1)
    boxBArea = (x22-x21+1) * (y22-y21+1)
    
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    
    return iou

if __name__=="__main__":

    test_tensor = torch.tensor([[10,10,20,20,0.4,0],[11,11,21,21,0.8,1]], dtype=torch.float32)
    print("Testing with input:")
    print(test_tensor)
    out_tensor = merge_bboxes(test_tensor, 0, 1, 2, 0.1)
    print("Output:")
    print(out_tensor)
    assert torch.equal(out_tensor,torch.tensor([[10,10,21,21,0.6,2]], dtype=torch.float32))
    print("Success")
    

        



