using Emgu.CV;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using System;
using System.Windows.Forms;

namespace MyEmguCVExample
{
    public partial class Form1 : Form
    {
        string imagePath = "";
        Image<Bgr, byte> image; 
        bool isCanny = false;

        public Form1()
        {
            InitializeComponent();
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            // 顯示預設路徑。
            ofd.InitialDirectory = "./";
            // 設定檔案儲存類型範例為.bmp若要增加其他類型可使用{名稱}|.{類型}。
            ofd.Filter = "Bitmap Files|*.bmp";
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
                image = new Image<Bgr, byte>(path);
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
        private void CaptureFace()
        {
            var umat = image.ToUMat();

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


            // 在影像上進行框線的繪圖
            for (int f = 0; f < faces.Count; f++)
                CvInvoke.Rectangle(umat, faces[f], new Bgr(Color.Red).MCvScalar, 2);
  
            imageBox1.Image = umat;

        }


        private void button3_Click(object sender, EventArgs e)
        {
            CaptureFace();
        }

        private void imageBox1_Click(object sender, EventArgs e)
        {

        }
    }
}