import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

import weka.core.Instance;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;

import java.io.FileWriter;

public class IragarpenakEgin {
    public static void main(String[] args) throws Exception {
        //-----------------------------FITXATEGIAK--------------------------------
        // Eredua kargatu
        NaiveBayes model = (NaiveBayes) weka.core.SerializationHelper.read(args[0]);

        // Test datuak kargatu
        DataSource source = new DataSource(args[1]);
        Instances testInstances = source.getDataSet();
        testInstances.setClassIndex(testInstances.numAttributes() - 1);

        // Emaitza gorde
        FileWriter fw = new FileWriter(args[2]);

        //-----------------------------IRAGARPENAK---------------------------------
        // Iragarpenak egin
        Evaluation evaluation = new Evaluation(testInstances);
        evaluation.evaluateModel(model, testInstances);

        //-------------------------------IDATZI------------------------------------
        // Iragarpenak gorde fitxategian
        for (Instance instance : testInstances) {
            int i = (int) model.classifyInstance(instance);
            fw.write(instance.toString() + ": " + testInstances.classAttribute().value(i) + "\n");
        }
    }
}

