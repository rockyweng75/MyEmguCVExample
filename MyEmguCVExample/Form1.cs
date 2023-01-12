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
            // ��ܹw�]���|�C
            ofd.InitialDirectory = "./";
            // �]�w�ɮ��x�s�����d�Ҭ�.bmp�Y�n�W�[��L�����i�ϥ�{�W��}|.{����}�C
            ofd.Filter = "Bitmap Files|*.bmp";
            ofd.Title = "��ܼv���ɮ�";
            // �����̫�������|��m�C
            ofd.RestoreDirectory = true;
            // ��ܵ����A����U�T�w(OK)�ɶi�J�C
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                // �O�s�ϥΪ̿���ɮת����|��m
                imagePath = ofd.FileName;
                textBox1.Text = imagePath;
                // Ū���v������ܡC
                DisplayImage(imagePath);
            }
        }

        private void DisplayImage(string path)
        {
            try
            {
                // �� Bitmap Ū���覡�C
                // Bitmap bitmap = new Bitmap(m_imagePath);
                // �Ӥ�k�h�� EmguCV ��Ū���覡�C
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
        /// �z�LOpenCV�i��H�y�O�_�s�b������
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


            // �b�v���W�i��ؽu��ø��
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