using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using System.Drawing;
using static System.Net.Mime.MediaTypeNames;
using System.Reflection.Emit;

namespace FaceDetectionLibrary
{
    public class FaceDetectionService : IDisposable
    {
        private EigenFaceRecognizer? faceRecognizer;

        /// <summary>
        /// 尚未載入學習資料，無法進行比對
        /// </summary>
        public bool IsWorking = false;

        public string trainfilepath = "";
        private bool disposedValue;

        public FaceDetectionService()
        {
            this.faceRecognizer = new EigenFaceRecognizer(0, double.PositiveInfinity);
        }

        public FaceDetectionService(EigenFaceRecognizer faceRecognizer) 
        {
            this.faceRecognizer = faceRecognizer;
        }

        public FaceDetectionService(EigenFaceRecognizer faceRecognizer, string trainfilepath)
        {
            this.faceRecognizer = faceRecognizer;
            this.faceRecognizer.Read(trainfilepath);
            this.IsWorking = true;
        }

        public void Read(string trainfilepath) 
        {
            faceRecognizer!.Read(trainfilepath);
            IsWorking = true;
        }

        public void Write(string trainfilepath)
        {
            faceRecognizer!.Write(trainfilepath);
        }


        public Image<Gray, Byte> CatchGrayFace(UMat umat, List<Rectangle>? faces = null, int width = 100, int heigth = 100) 
        {
            if (faces == null) 
            {
                faces = CaptureFace(umat);
            }

            if (faces.Count == 0) throw new Exception("找不到臉部特徵");

            return umat
                    .ToImage<Gray, Byte>()
                    .GetSubRect(faces.First())
                    .Resize(width, heigth, Inter.Cubic);
        }

        public Image<Bgr, Byte> CatchBgrFace(UMat umat, List<Rectangle>? faces = null, int width = 100, int heigth = 100)
        {
            if (faces == null)
            {
                faces = CaptureFace(umat);
            }

            if (faces.Count == 0) throw new Exception("找不到臉部特徵");

            return umat
                    .ToImage<Bgr, Byte>()
                    .GetSubRect(faces.First())
                    .Resize(width, heigth, Inter.Cubic);
        }

        /// <summary>
        /// 透過OpenCV進行眼睛是否存在的辨識
        /// </summary>
        /// <param name="objMat"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public List<Rectangle> CaptureEyes(UMat umat)
        {
            List<Rectangle> faces = new List<Rectangle>();

            using (CascadeClassifier faceDetector = new CascadeClassifier("./haarcascade_eye.xml"))
            {
                using (UMat ugray = new UMat())
                {
                    CvInvoke.CvtColor(umat, ugray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

                    //normalizes brightness and increases contrast of the image
                    CvInvoke.EqualizeHist(ugray, ugray);

                    //Detect the faces  from the gray scale image and store the locations as rectangle                   
                    Rectangle[] facesDetected = faceDetector.DetectMultiScale(
                       ugray, 1.1, 10, new Size(20, 20));

                    faces.AddRange(facesDetected);
                }
            }
            return faces;
        }


        /// <summary>
        /// 透過OpenCV進行人臉是否存在的辨識
        /// </summary>
        /// <param name="objMat"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public List<Rectangle> CaptureFace(UMat umat)
        {
            List<Rectangle> faces = new List<Rectangle>();

            using (CascadeClassifier faceDetector = new CascadeClassifier("./haarcascade_frontalface_default.xml"))
            {
                using (UMat ugray = new UMat())
                {
                    CvInvoke.CvtColor(umat, ugray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

                    //normalizes brightness and increases contrast of the image
                    CvInvoke.EqualizeHist(ugray, ugray);

                    //Detect the faces  from the gray scale image and store the locations as rectangle                   
                    Rectangle[] facesDetected = faceDetector.DetectMultiScale(
                       ugray, 1.1, 10, new Size(20, 20));

                    faces.AddRange(facesDetected);
                }
            }
            return faces;
        }

        /// <summary>
        /// 在影像上進行框線的繪圖
        /// </summary>
        /// <param name="umat"></param>
        /// <param name="faces"></param>
        /// <returns></returns>
        public UMat PrintFaces(UMat umat, List<Rectangle> faces)
        {
            return PrintBorder(umat, faces);
        }

        public UMat PrintEyes(UMat umat, List<Rectangle> eyes)
        {
            return PrintBorder(umat, eyes);
        }

        private UMat PrintBorder(UMat umat, List<Rectangle> ractangles) 
        {
            for (int f = 0; f < ractangles.Count; f++)
                CvInvoke.Rectangle(umat, ractangles[f], new Bgr(Color.Red).MCvScalar, 2);
            return umat;
        }

        /// <summary>
        /// 比對臉部影像，取回最接近的影像編號
        /// </summary>
        /// <param name="faceImage"></param>
        /// <returns>
        ///     Item1: 分數
        ///     Item2: 接近的影像編號
        ///     Item3: 執行成功
        /// </returns>
        public (double, int, bool) FaceRecognizer(Image<Gray, Byte> faceImage)
        {
            if (IsWorking)
            {
                faceImage = faceImage.Resize(100, 100, Inter.Cubic);
                var predictResult = faceRecognizer!.Predict(faceImage);
                return (predictResult.Distance, predictResult.Label, true);
            }
            return (-1, -1, false);

        }
        /// <summary>
        /// 多筆訓練
        /// </summary>
        /// <param name="images"></param>
        /// <param name="labels"></param>
        /// <returns></returns>
        public bool BatchTrain(IList<Image<Bgr, Byte>> images, int[] labels)
        {
            var faces = new List<Mat>();
            foreach (var image in images) 
            {
                faces.Add(CatchGrayFace(image.ToUMat()).Mat);
            }

            //進行訓練
            faceRecognizer!.Train(faces.ToArray(), labels);

            return true;
        }

        /// <summary>
        /// 單筆訓練
        /// </summary>
        /// <param name="image"></param>
        /// <param name="label"></param>
        /// <returns></returns>
        public bool Train(Image<Bgr, Byte> image, int label)
        {
            var faces = new List<Mat>();

            faces.Add(CatchGrayFace(image.ToUMat()).Mat);

            var labels = new List<int>() { label };

            //進行訓練
            faceRecognizer!.Train(faces.ToArray(), labels.ToArray());

            return true;
        }
    

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    faceRecognizer!.Dispose();
                }

                // TODO: 釋出非受控資源 (非受控物件) 並覆寫完成項
                // TODO: 將大型欄位設為 Null
                disposedValue = true;
            }
        }

        // // TODO: 僅有當 'Dispose(bool disposing)' 具有會釋出非受控資源的程式碼時，才覆寫完成項
        // ~FaceDetection()
        // {
        //     // 請勿變更此程式碼。請將清除程式碼放入 'Dispose(bool disposing)' 方法
        //     Dispose(disposing: false);
        // }

        void IDisposable.Dispose()
        {
            // 請勿變更此程式碼。請將清除程式碼放入 'Dispose(bool disposing)' 方法
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}