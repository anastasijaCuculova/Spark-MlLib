import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.*;

/*
 * The goal is to divide data in two data sets training and test set.
 * After that data must be parsed in LabeledPoint and given to the algorithm to predict the classifier
 * */
public class WineClassification {

    private final static int SEVENTY_PERCENTS_OF_DATA = 70;
    private final static int THIRTY_PERCENTS_OF_DATA = 30;
    private final static String PATH_TO_FILE = "C:\\Users\\anastasija.cuculova\\workspace\\spark-ml-course\\src\\main\\resources\\wine.data";

    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("Wine App").setMaster("local[*]");
        JavaSparkContext context = new JavaSparkContext(sparkConf);
        SparkSession session = SparkSession.builder().getOrCreate();
        Dataset<Row> dataSet = session.read().option("header", true)
                .csv(PATH_TO_FILE);
        List<LabeledPoint> data = loadData(context);

        // 70% for training data
        // 30% for test data

        int numberOfElementsInTrainingData = ((data.size() - 1) * SEVENTY_PERCENTS_OF_DATA) / 100;
        int numberOfElementsInTestData = ((data.size() - 1) * THIRTY_PERCENTS_OF_DATA) / 100;
        Object[] trainingData = data.subList(0, numberOfElementsInTrainingData).toArray();
        Object[] testData = data.subList(numberOfElementsInTestData, (data.size() - 1)).toArray();

        double[] testDoubleData = parseDataToDouble(testData);
        List<LabeledPoint> trainingParsedData = parseData(trainingData);
        List<LabeledPoint> testParsedData = parseData(testData);
        prepareDecisionTreeAlgorithm(context.parallelize(trainingParsedData), context.parallelize(testParsedData), testDoubleData, dataSet);
    }

    private static List<LabeledPoint> loadData(JavaSparkContext context) {
        JavaRDD<String> javaRDD =
                context.textFile(PATH_TO_FILE);
        List<String[]> parts = javaRDD.map(line -> line.split(",")).collect();

        List<LabeledPoint> parsedDataAsList = new ArrayList<>();
        for (String[] part : parts) {
            double[] doubleValues = new double[part.length];
            for (int i = 0; i < part.length; i++) {
                if (part[i] != null) {
                    doubleValues[i] = new Double(part[i]);
                }
            }
            parsedDataAsList.add(new LabeledPoint(doubleValues[0], Vectors.dense(doubleValues)));
        }
        return parsedDataAsList;
    }

    private static double[] parseDataToDouble(Object[] data) {
        double[] doubleValues = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            if (data[i] instanceof Double) {
                doubleValues[i] = new Double(data[i].toString());
            }
        }
        return doubleValues;
    }

    private static void prepareDecisionTreeAlgorithm(JavaRDD<LabeledPoint> trainingParsedData,
                                                     JavaRDD<LabeledPoint> testParsedData,
                                                     double[] testDoubleData,
                                                     Dataset<Row> dataSet) {
        // Set parameters.
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        int numClasses = 3;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "gini";
        int maxDepth = 5;
        int maxBins = 32;
        DecisionTreeModel model = DecisionTree.trainClassifier(trainingParsedData, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);
        double predictValue = model.predict(Vectors.dense(testDoubleData));
        System.out.println("Predicted value: " + predictValue);
    }

    private static List<LabeledPoint> parseData(Object[] data) {
        LabeledPoint[] result = new LabeledPoint[data.length];
        for (int i = 0; i < data.length; i++) {
            if (data[i] instanceof LabeledPoint) {
                result[i] = (LabeledPoint) data[i];
            }
        }
        return Arrays.asList(result);
    }
}
