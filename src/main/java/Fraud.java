import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;

import java.util.Arrays;
import java.util.stream.Stream;

public class Fraud {
    public static void main(String[] args) {
        AlgorithmProperties configuration = new AlgorithmProperties();
        SparkConf sparkConf = new SparkConf().setAppName("Fraud").setMaster("local[*]");
        JavaSparkContext context = new JavaSparkContext(sparkConf);
        JavaRDD<LabeledPoint> labeledPointJavaRDD = loadAndParseData(configuration, context);

        JavaRDD<LabeledPoint> anomalies = filterAnomalies(labeledPointJavaRDD);
        JavaRDD<LabeledPoint> regularData = filterRegularData(labeledPointJavaRDD);
        System.out.println(labeledPointJavaRDD.collect());
        System.out.println(anomalies.collect());
        System.out.println(regularData.collect());
    }

    private static JavaRDD<LabeledPoint> filterRegularData(JavaRDD<LabeledPoint> labeledPointJavaRDD) {
        return labeledPointJavaRDD.filter(e -> e.label() == 0d);
    }

    private static JavaRDD<LabeledPoint> filterAnomalies(JavaRDD<LabeledPoint> labeledPointJavaRDD) {
        return labeledPointJavaRDD.filter(e -> e.label() == 1d);
    }


    private static JavaRDD<LabeledPoint> loadAndParseData(AlgorithmProperties configuration, JavaSparkContext sparkContext) {
        // Because all data should be normalized, in my case all columns will have numeric values
        return sparkContext
                .textFile("C:\\Users\\anastasija.cuculova\\workspace\\spark-ml-course\\src\\main\\resources\\transactions.csv")
                .map(line -> {
                            line = line.replace(TransactionType.PAYMENT.name(), "1")
                                    .replace(TransactionType.TRANSFER.name(), "2")
                                    .replace(TransactionType.CASH_OUT.name(), "3")
                                    .replace(TransactionType.DEBIT.name(), "4")
                                    .replace(TransactionType.CASH_IN.name(), "5")
                                    .replace("C", "1")
                                    .replace("M", "2");
                            String[] split = line.split(",");
                            //skip header
                            if (split[0].equalsIgnoreCase("step")) {
                                return null;
                            }
                            double[] featureValues = Stream.of(split)
                                    .mapToDouble(Double::parseDouble).toArray();
                            if (configuration.isMakeFeaturesMoreGaussian()) {
                                makeFeaturesMoreGaussian(featureValues);
                            }
                            double label = featureValues[9];
                            featureValues = Arrays.copyOfRange(featureValues, 0, 9);
                            return new LabeledPoint(label, Vectors.dense(featureValues));
                        }
                ).cache();
    }

    private static void makeFeaturesMoreGaussian(double[] featureValues) {
        double[] powers = {0.5, 1, 0.1, 0.3, 0.1, 0.08, 0.3, 0.1, 0.1, 1, 1};
        for (int i = 0; i < featureValues.length; i++) {
            featureValues[i] = Math.pow(featureValues[i], powers[i]);
        }
    }
}
