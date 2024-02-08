import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import java.util.Random;
import java.io.*;


public class ModeloaSortu {
    public static void main(String[] args) {
        try {
            // Fitxategiak
            String inPath = args[0];
            DataSource source = new DataSource(inPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            String outPath = args[1];
            FileWriter fw = new FileWriter(outPath);

            String modelPath = args[2];

            // NaiveBayes eredua entrenatu
            NaiveBayes model = new NaiveBayes();
            model.buildClassifier(data);
            weka.core.SerializationHelper.write(modelPath, model); //gorde

            // K-fcv
            Evaluation kfcv = new Evaluation(data);
            kfcv.crossValidateModel(model, data, 5, new Random(1));

            // Hold-out
                // banaketa
            int trainSize = (int) Math.round(data.numInstances() * 0.7);
            int testSize = data.size() - trainSize;
            Instances trainData = new Instances(data, 0, trainSize);
            Instances testData = new Instances(data, trainSize, testSize);
                // ebaluaketa
            model.buildClassifier(trainData);
            Evaluation holdOut = new Evaluation(data);
            holdOut.evaluateModel(model, testData);

            // Emaitzak gorde
            fw.write("K-fcv:\n" + kfcv.toMatrixString() + "\n");
            fw.write("Hold-out:\n" + holdOut.toMatrixString() + "\n");

            //fileWriter itxi
            fw.flush();
            fw.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

