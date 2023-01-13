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
        /// �z�LOpenCV�i��H�y�O�_�s�b������
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
            // �b�v���W�i��ؽu��ø��
            for (int f = 0; f < faces.Count; f++)
                CvInvoke.Rectangle(umat, faces[f], new Bgr(Color.Red).MCvScalar, 2);

            imageBox1.Image = umat;
        }

        private (double, int, bool) FaceRecognizer(List<Rectangle> faces) 
        {

            FileStream fileStream = new FileStream(sourcePath, FileMode.Open, FileAccess.Read);

            //�����O�_���H�y
            if (faces.Count > 0) 
            {
                using (var faceRecognizer = new EigenFaceRecognizer(0, double.PositiveInfinity))
                {
                    faceRecognizer.Read(sourcePath);

                    foreach (var item in faces)
                    {
                        //�v��Ϥ��Ƕ��� �ó]�w�j�p100
                        var ToGrayFace = image
                            .Convert<Gray, Byte>()
                            .GetSubRect(item)
                            .Resize(100, 100, Inter.Cubic);

                        if (fileStream.Length != 0)
                        {
                            //���s�ŧi
                            var predictResult = faceRecognizer.Predict(ToGrayFace);

                            //���@����Ƨk�X ��L�h���A����
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
            //faceRecognizer = new EigenFaceRecognizer(80, double.PositiveInfinity);

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
            var faces = CaptureFace(umat);

            PrintFace(umat, faces);
            //var result = FaceRecognizer(faces);
            //label1.Text = $"��ﵲ�G => ����:{result.Item1}, �ӷ��s���G{result.Item2}, ���G�G{result.Item3}";

        }

        private void button4_Click(object sender, EventArgs e)
        {
            FaceTraining();
            label1.Text = $"�ǲߧ���";
        }

        private void button5_Click(object sender, EventArgs e)
        {
            var umat = image.ToUMat();
            var faces = CaptureFace(umat);
            var result = FaceRecognizer(faces);
            label1.Text = $"��ﵲ�G => ����:{result.Item1}, �ӷ��s���G{result.Item2}, ���G�G{result.Item3}";

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
                    label1.Text = $"����:{result.Distance}";
                }
            }
        }

        private void imageBox1_Click(object sender, EventArgs e)
        {

        }
    }
}