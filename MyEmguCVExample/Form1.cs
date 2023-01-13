using Emgu.CV;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using System;
using System.Windows.Forms;
using Emgu.CV.Face;
using Emgu.CV.CvEnum;

namespace MyEmguCVExample
{
    public partial class Form1 : Form
    {
        string imagePath = "";
        Image<Bgr, byte> image;
        bool isCanny = false;
        double faceBase = 160;
        string sourcePath = "faces.dat";
        string imagesFolder = "./Data";

        // https://thispersondoesnotexist.com/

        public Form1()
        {
            InitializeComponent();


            if (!File.Exists(sourcePath))
            {
                File.Create(sourcePath);
            }
            else 
            {
                //File.Delete(sourcePath);
                //File.Create(sourcePath);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            // 顯示預設路徑。
            ofd.InitialDirectory = "./";
            // 設定檔案儲存類型範例為.bmp若要增加其他類型可使用{名稱}|.{類型}。
            ofd.Filter = "Bitmap Files|*.bmp|jpg|*.jpg";
            ofd.Title = "選擇影像檔案";
            // 紀錄最後視窗路徑位置。
            ofd.RestoreDirectory = true;
            // 顯示視窗，當按下確定(OK)時進入。
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                // 保存使用者選擇檔案的路徑位置
                imagePath = ofd.FileName;
                textBox1.Text = imagePath;
                // 讀取影像並顯示。
                DisplayImage(imagePath);
            }
        }

        private void DisplayImage(string path)
        {
            try
            {
                // 為 Bitmap 讀取方式。
                // Bitmap bitmap = new Bitmap(m_imagePath);
                // 該方法則為 EmguCV 的讀取方式。
                image = new Image<Bgr, byte>(path).Resize(300, 300, Inter.Cubic);
                pictureBox1.Image = (Bitmap)image.ToBitmap().Clone();
                pictureBox1.Invalidate();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {
            if (!isCanny)
            {
                var frame = pictureBox1.Image;
                Image<Gray, Byte> grayFrame = image.Convert<Gray, Byte>();
                Image<Gray, Byte> cannyFrame = grayFrame.Canny(100, 60);
                pictureBox1.Image = (Bitmap)cannyFrame.ToBitmap().Clone();
                isCanny = true;
            }
            else
            {
                pictureBox1.Image = (Bitmap)image.ToBitmap().Clone();
                isCanny = false;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            pictureBox1.Image = null;
        }

        /// <summary>
        /// 透過OpenCV進行人臉是否存在的辨識
        /// </summary>
        /// <param name="objMat"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        private List<Rectangle> CaptureFace(UMat umat)
        {
            List<Rectangle> faces = new List<Rectangle>();

            using (CascadeClassifier faceDetector = new CascadeClassifier("./haarcascade_frontalface_default.xml"))
            {

                using (UMat ugray = new UMat())
                {

                    CvInvoke.CvtColor(umat, ugray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

                    //imageBox1.Image = ugray;

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


        private void PrintFace(UMat umat, List<Rectangle> faces)
        {
            // 在影像上進行框線的繪圖
            for (int f = 0; f < faces.Count; f++)
                CvInvoke.Rectangle(umat, faces[f], new Bgr(Color.Red).MCvScalar, 2);

            imageBox1.Image = umat;
        }

        private (double, int, bool) FaceRecognizer(List<Rectangle> faces) 
        {

            FileStream fileStream = new FileStream(sourcePath, FileMode.Open, FileAccess.Read);

            //偵測是否為人臉
            if (faces.Count > 0) 
            {
                using (var faceRecognizer = new EigenFaceRecognizer(0, double.PositiveInfinity))
                {
                    faceRecognizer.Read(sourcePath);

                    foreach (var item in faces)
                    {
                        //影格圖片灰階化 並設定大小100
                        var ToGrayFace = image
                            .Convert<Gray, Byte>()
                            .GetSubRect(item)
                            .Resize(100, 100, Inter.Cubic);

                        if (fileStream.Length != 0)
                        {
                            //重新宣告
                            var predictResult = faceRecognizer.Predict(ToGrayFace);

                            //當有一筆資料吻合 其他則不再辨識
                            if (predictResult.Distance <= faceBase)
                            {
                                faceRecognizer.Dispose();
                                faceRecognizer = null;
                                return (predictResult.Distance, predictResult.Label, true);
                            }
                        }
                    }
                }
            }
            return (0, 0, false);

        }

        /// <summary>
        /// 單次訓練
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public int FaceTraining()
        {
            var facesMat = new List<Mat>();
            var labels = new List<int>();

            //取得路徑下所有照片
            int PictureCount = 0;
            //faceRecognizer = new EigenFaceRecognizer(80, double.PositiveInfinity);

            using (var faceRecognizer = new EigenFaceRecognizer(0, double.PositiveInfinity))
            {

                //照片灰階處理
                facesMat.Add(image.Convert<Gray, Byte>().Resize(100, 100, Inter.Cubic).Mat);
                //照片Tag
                labels.Add(1);
                PictureCount += 1;

                //進行訓練
                faceRecognizer.Train(facesMat.ToArray(), labels.ToArray());
                faceRecognizer.Write(sourcePath);
            }
            return PictureCount;
        }

        private void button3_Click(object sender, EventArgs e)
        {

            var umat = image.ToUMat();
            var faces = CaptureFace(umat);

            PrintFace(umat, faces);
            //var result = FaceRecognizer(faces);
            //label1.Text = $"比對結果 => 分數:{result.Item1}, 來源編號：{result.Item2}, 結果：{result.Item3}";

        }

        private void button4_Click(object sender, EventArgs e)
        {
            FaceTraining();
            label1.Text = $"學習完成";
        }

        private void button5_Click(object sender, EventArgs e)
        {
            var umat = image.ToUMat();
            var faces = CaptureFace(umat);
            var result = FaceRecognizer(faces);
            label1.Text = $"比對結果 => 分數:{result.Item1}, 來源編號：{result.Item2}, 結果：{result.Item3}";

            umat.Dispose();

        }

        private void button6_Click(object sender, EventArgs e)
        {
            label1.Text = "";
            List <Mat> trainingImages = new List<Mat>();
            List<int> labels = new List<int>();
            List<ImageBox> imageBoxes = new List<ImageBox>() { null, imageBox1, imageBox2, imageBox3};
            for (var i = 1; i <= 3; i++) 
            {
                var _image = new Image<Bgr, byte>($"{imagesFolder}/{i}.bmp")
                    .Resize(300, 300, Inter.Cubic);

                var faces = CaptureFace(_image.ToUMat());

                foreach (var item in faces) 
                {
                    var ToGrayFace = _image
                       .Convert<Gray, Byte>()
                       .GetSubRect(item)
                       .Resize(100, 100, Inter.Cubic);

                    trainingImages.Add(ToGrayFace.Mat);
                    imageBoxes[i].Image = ToGrayFace;
                }

                labels.Add(i);
            }
            //MCvTermCriteria termCrit = new MCvTermCriteria(16, 0.001);

            using (EigenFaceRecognizer recognizer = new EigenFaceRecognizer(80, double.PositiveInfinity))
            {
                recognizer.Train(trainingImages.ToArray(), labels.ToArray());
                recognizer.Write(sourcePath);

            }

            using (EigenFaceRecognizer recognizer = new EigenFaceRecognizer(80, double.PositiveInfinity)) 
            {
                recognizer.Read(sourcePath);

                var testImage = new Image<Bgr, Byte>($"{imagesFolder}/2.bmp")
                    .Resize(300, 300, Inter.Cubic);

                var faces = CaptureFace(testImage.ToUMat());

                foreach (var item in faces)
                {
                    var ToGrayFace = testImage
                      .Convert<Gray, Byte>()
                      .GetSubRect(item)
                      .Resize(100, 100, Inter.Cubic);

                    EigenFaceRecognizer.PredictionResult result = recognizer.Predict(ToGrayFace);
                    label1.Text = $"分數:{result.Distance}";
                }
            }
        }

        private void imageBox1_Click(object sender, EventArgs e)
        {

        }
    }
}