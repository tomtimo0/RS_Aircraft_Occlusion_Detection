# E:\RS_Aircraft_Occlusion_Detection\ImgSplit.py 文件内容 (已更新)

import os
import codecs
import numpy as np
import math
import cv2
import shapely.geometry as shgeo
import copy

try:
    import dota_utils as util
    from dota_utils import GetFileFromThisRootDir
except ImportError:
    print("错误：无法找到 'dota_utils.py' 文件。")
    print("请确保 'dota_utils.py' 与 ImgSplit.py 在同一个文件夹中。")
    exit()

def choose_best_pointorder_fit_another(poly1, poly2):
    x1, y1, x2, y2, x3, y3, x4, y4 = poly1[0], poly1[1], poly1[2], poly1[3], poly1[4], poly1[5], poly1[6], poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    return combinate[distances.argsort()[0]]

def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code='utf-8',
                 gap=512,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext='.png',
                 padding=True,
                 #  <<<<<<<<<<<<<<<<<< 新增参数：允许自定义标签文件夹名称 >>>>>>>>>>>>>>>>
                 label_dirname='labelTxt'
                 ):
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, 'images')
        #  <<<<<<<<<<<<<<<<<< 使用新参数来构建标签路径 >>>>>>>>>>>>>>>>
        self.labelpath = os.path.join(self.basepath, label_dirname)
        self.outimagepath = os.path.join(self.outpath, 'images')
        self.outlabelpath = os.path.join(self.outpath, 'labelTxt')
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding
        print('是否进行填充 (padding):', padding)
        print(f"图像路径: {self.imagepath}")
        print(f"标签路径: {self.labelpath}") # 打印路径方便调试

        # 检查标签路径是否存在
        if not os.path.isdir(self.labelpath):
            print(f"错误：找不到标签路径 {self.labelpath}")
            print("请检查 `base_path` 和 `label_folder_name` 是否设置正确。")
            exit() # 如果路径不存在则直接退出

        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        if not os.path.isdir(self.outimagepath):
            os.mkdir(self.outimagepath)
        if not os.path.isdir(self.outlabelpath):
            os.mkdir(self.outlabelpath)
            
    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        inter_poly = poly1.intersection(poly2)
        return inter_poly, inter_poly.area / poly1.area

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3), dtype=np.uint8)
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(4)]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            if count == pos:
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
                outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
                count += 1
            elif count == (pos + 1)%5:
                count += 1
                continue
            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count += 1
        return outpoly

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down), (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                if len(obj['poly']) != 8:
                    continue
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]), (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]), (obj['poly'][6], obj['poly'][7])])
                if gtpoly.area <= 0:
                    continue
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

                if half_iou == 1:
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ' '.join(map(str, polyInsub)) + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                elif half_iou > 0:
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[:-1]
                    if len(out_poly) < 4:
                        continue
                    
                    out_poly2 = [p for xy in out_poly for p in xy]

                    if len(out_poly) == 5:
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                    elif len(out_poly) > 5:
                        continue
                    if self.choosebestpoint:
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)
                    for i, item in enumerate(polyInsub):
                        polyInsub[i] = min(max(item, 1), self.subsize)
                    
                    outline = ' '.join(map(str, polyInsub))
                    if half_iou > self.thresh:
                        outline += ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        outline += ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')
        self.saveimagepatches(resizeimg, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        img_path = os.path.join(self.imagepath, name + extent)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像 {img_path}")
            return
        
        label_path = os.path.join(self.labelpath, name + '.txt')
        if not os.path.exists(label_path):
            # 这条警告现在应该不会出现了
            print(f"警告：找不到对应的标签文件 {label_path}")
            objects = []
        else:
            objects = util.parse_dota_poly2(label_path)

        for obj in objects:
            obj['poly'] = [p * rate for p in obj['poly']]

        resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC) if rate != 1 else img
        
        outbasename = f"{name}__{rate}__"
        height, width = resizeimg.shape[:2]

        left, up = 0, 0
        while left < width:
            if left + self.subsize >= width:
                left = max(width - self.subsize, 0)
            up = 0
            while up < height:
                if up + self.subsize >= height:
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, width - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = f"{outbasename}{left}___{up}"
                self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                if up + self.subsize >= height:
                    break
                up += self.slide
            if left + self.subsize >= width:
                break
            left += self.slide

    def splitdata(self, rate):
        imagelist = GetFileFromThisRootDir(self.imagepath)
        imagenames = [util.custombasename(x) for x in imagelist if util.custombasename(x) != 'Thumbs']
        for i, name in enumerate(imagenames):
            print(f'正在处理 {i+1}/{len(imagenames)}: {name}')
            self.SplitSingle(name, rate, self.ext)

if __name__ == '__main__':
    # ======================= 参数配置区域 =======================
    
    base_path = r'E:\RS_Aircraft_Occlusion_Detection\dataset\train_test'
    output_path = r'E:\RS_Aircraft_Occlusion_Detection\dataset\train_split_test'
    
    #  <<<<<<<<<<<<<<<<<< 在这里指定你真实的标签文件夹名称 >>>>>>>>>>>>>>>>
    label_folder_name = 'labelTxt-v2.0' 
    
    subimage_size = 1024
    overlap = 200
    image_extension = '.png'
    
    split = splitbase(
        basepath=base_path,
        outpath=output_path,
        subsize=subimage_size,
        gap=overlap,
        ext=image_extension,
        #  <<<<<<<<<<<<<<<<<< 将文件夹名称传递给类 >>>>>>>>>>>>>>>>
        label_dirname=label_folder_name
    )

    print(f"\n开始切割，缩放比例: 1.0 (原始尺寸)...")
    split.splitdata(1.0)
    print("缩放比例 1.0 的数据切割完成。")

    print(f"\n所有切割任务已完成！")
    print(f"切割后的数据已保存至: {output_path}")
    
    # ======================== 配置结束 =========================