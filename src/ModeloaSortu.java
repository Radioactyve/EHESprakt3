import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;

import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.util.Random;
import java.io.*;


public class ModeloaSortu {
    public static void main(String[] args) {
        try {
            // ------------------------FITXATEGIAK-----------------------------
            String inPath = args[0];
            DataSource source = new DataSource(inPath);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            String outPath = args[1];
            FileWriter fw = new FileWriter(outPath);

            String modelPath = args[2];

            // ------------------------MODELOA SORTU---------------------------
            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(data); //eredua entrenatu
            weka.core.SerializationHelper.write(modelPath, nb); //gorde


            // ---------------------------EBALUATU------------------------------
            // K-FCV
            Evaluation kfcv = new Evaluation(data);
            kfcv.crossValidateModel(nb, data, 5, new Random(1));

            // HOLD-OUT
            // banaketa
            data.randomize(new java.util.Random(1));
            RemovePercentage f1 = new RemovePercentage();
            f1.setPercentage(0.7);

            f1.setInputFormat(data);
            f1.setInvertSelection(true);
            Instances trainData = Filter.useFilter(data, f1); //train

            f1.setInputFormat(data);
            f1.setInvertSelection(false);
            Instances testData = Filter.useFilter(data, f1); //test
            // ebaluaketa
            nb.buildClassifier(trainData);
            Evaluation holdOut = new Evaluation(data);
            holdOut.evaluateModel(nb, testData);

            //-----------------------------IDATZI------------------------------
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

