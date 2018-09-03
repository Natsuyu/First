import torch
import pickle
import os
import cv2

def touch_dir(path):
    result = False
    try:
        path = path.strip().rstrip("\\")
        if not os.path.exists(path):
            os.makedirs(path)
            result = True
        else:
            result = True
    except:
        result = False
    return result

def load_test_data():
    path = '../input/test_data.pkl' # to be confirmed
    test_arr = pickle.load(open(path,'rb'))
    return test_arr


if __name__ == '__main__':
    path = '../input/train_model.pkl'
    model = pickle.load(open(path, 'rb'))
    # test_arr = load_test_data()

    with torch.no_grad():
        # model.eval()
        path = '../input/SRAD2018_Test'
        files = os.listdir(path)
        s = []
        cnt = 0
        for file in files:
            cnt += 1
            img = []
            t_path = path + '/' + file
            t_files = os.listdir(t_path)
            for f in t_files:
                if not os.path.isdir(t_path + '/' + f):
                    if f == '.DS_Store': continue
                    tmp = cv2.imread(t_path + '/' + f)

                    img.append(tmp)

            #             print img
            img = torch.FloatTensor(img)

            save_path = '../output/result'
            touch_dir(save_path)

            input_image = img / 255.

            input_gru = input_image.cuda()
            #          target_gru = target_image.cuda()

            fx = model.forward(input_gru)
            for pre_id in range(len(fx)):
                temp_xx = fx[pre_id].cpu().data.numpy()
                tmp_img = temp_xx[0, 0, ...]
                tmp_img = tmp_img * 255.
                #             true_img = target_image[pre_id, 0,  ...]
                #             encode_img = input_image[pre_id, 0, 0, ...]
                # cv2.imwrite(os.path.join(save_path, 'a_%s.png' % pre_id), encode_img)
                cv2.imwrite(os.path.join(save_path, 'RAD_%s_f00%s.png' % pre_id),t_path, pre_id)
                # cv2.imwrite(os.path.join(save_path, 'b_%s.png' % pre_id), true_img)

        # for pre_data in pre_list:
        #     temp = pre_data.cpu().data.numpy()
        #     print temp.mean()

