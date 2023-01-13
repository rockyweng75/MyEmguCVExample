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
            // ��ܹw�]���|�C
            ofd.InitialDirectory = "./";
            // �]�w�ɮ��x�s�����d�Ҭ�.bmp�Y�n�W�[��L�����i�ϥ�{�W��}|.{����}�C
            ofd.Filter = "Bitmap Files|*.bmp|jpg|*.jpg";
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
        /// �榸�V�m
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public int FaceTraining()
        {
            var facesMat = new List<Mat>();
            var labels = new List<int>();

            //���o���|�U�Ҧ��Ӥ�
            int PictureCount = 0;

            using (var faceRecognizer = new EigenFaceRecognizer(0, double.PositiveInfinity))
            {

                //�Ӥ��Ƕ��B�z
                facesMat.Add(image.Convert<Gray, Byte>().Resize(100, 100, Inter.Cubic).Mat);
                //�Ӥ�Tag
                labels.Add(1);
                PictureCount += 1;

                //�i��V�m
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
                textBox2.Text = "�п�J�Ӥ��s��(�Ʀr)";
            }
            else if (int.TryParse(_label,  out var label)) 
            {
                faceDetectionService.Train(image, label);
                label1.Text = $"�ǲߧ���";
            }
            else
            {
                textBox2.Text = "�п�J�Ӥ��s��(�Ʀr)";
            }
        }

        private void button5_Click(object sender, EventArgs e)
        {
            var umat = image.ToUMat();
            var faces = faceDetectionService.CatchGrayFace(umat);
            var result = faceDetectionService.FaceRecognizer(faces);
            if (result.Item3)
            {
                label1.Text = $"��ﵲ�G => ����:{result.Item1}, �ӷ��s���G{result.Item2}, ���G�G{(result.Item1 <= faceBase ? "�q�L" : "���q�L")}";
            }
            else 
            {
                label1.Text = "�L�k����";
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
            label1.Text = $"���J���\�A���դ���:{result.Item1}";

        }

    }
}