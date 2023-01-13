using Emgu.CV;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using System;
using System.Windows.Forms;
using Emgu.CV.Face;
using Emgu.CV.CvEnum;
using FaceDetectionLibrary;

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

        FaceDetectionService faceDetectionService;
        // https://thispersondoesnotexist.com/

        public Form1(FaceDetectionService faceDetectionService)
        {
            InitializeComponent();

            this.faceDetectionService = faceDetectionService;

            if (!File.Exists(sourcePath))
            {
                File.Create(sourcePath);
            }

            if (!faceDetectionService.IsWorking){
                faceDetectionService.Read(sourcePath);
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
            var faces = faceDetectionService.CaptureFace(umat);
            faceDetectionService.PrintFaces(umat, faces);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            var _label = textBox2.Text;
            if (string.IsNullOrEmpty(_label))
            {
                textBox2.Text = "請輸入照片編號(數字)";
            }
            else if (int.TryParse(_label,  out var label)) 
            {
                faceDetectionService.Train(image, label);
                label1.Text = $"學習完成";
            }
            else
            {
                textBox2.Text = "請輸入照片編號(數字)";
            }
        }

        private void button5_Click(object sender, EventArgs e)
        {
            var umat = image.ToUMat();
            var faces = faceDetectionService.CatchGrayFace(umat);
            var result = faceDetectionService.FaceRecognizer(faces);
            if (result.Item3)
            {
                label1.Text = $"比對結果 => 分數:{result.Item1}, 來源編號：{result.Item2}, 結果：{(result.Item1 <= faceBase ? "通過" : "不通過")}";
            }
            else 
            {
                label1.Text = "無法辨識";
            }
            umat.Dispose();
        }

        private void button6_Click(object sender, EventArgs e)
        {

            label1.Text = "";
            List<Image<Bgr, byte>> trainingImages = new List<Image<Bgr, byte>>();
            List<int> labels = new List<int>();
            List<ImageBox> imageBoxes = new List<ImageBox>() { null, imageBox1, imageBox2, imageBox3};
            for (var i = 1; i <= 3; i++) 
            {
                var _image = new Image<Bgr, byte>($"{imagesFolder}/{i}.bmp")
                    .Resize(300, 300, Inter.Cubic);

                trainingImages.Add(_image);
                imageBoxes[i].Image = _image;
                labels.Add(i);
            }

            faceDetectionService.BatchTrain(trainingImages.ToArray(), labels.ToArray());
            faceDetectionService.Write(sourcePath);

            var testImage = new Image<Bgr, Byte>($"{imagesFolder}/2.bmp")
                .Resize(300, 300, Inter.Cubic);

            var faces = faceDetectionService.CatchGrayFace(testImage.ToUMat());

            var result = faceDetectionService.FaceRecognizer(faces);
            label1.Text = $"載入成功，測試分數:{result.Item1}";

        }

    }
}