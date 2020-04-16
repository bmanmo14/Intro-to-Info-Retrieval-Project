import java.util.HashMap;
import java.util.Map;

public class ModelMixer /*extends Thread*/ {
    HashMap<String, Double> docs1;
    HashMap<String, Double> docs2;
    QueryResultsIO writer;
    String qid;
    double alpha;
    int docsRequested;
//    HashMap<String, HashMap<String, Double>> allScores;

    ModelMixer(HashMap<String, Double> docs1,
               HashMap<String, Double> docs2,
               QueryResultsIO writer,
               String qid,
               double alpha,
               int docsRequested) {

//        this.allScores = allScores;
        this.docs1 = docs1;
        this.docs2 = docs2;
        this.writer = writer;
        this.qid = qid;
        this.alpha = alpha;
        this.docsRequested = docsRequested;
//        start();
        normalize(docs1, docs2);
        HashMap<String, Double> results = combine(docs1, docs2, alpha);
        HashMap<String, Double> sortedResults = QueryResultsIO.sortByValue(results);
        writer.writeLines(qid, sortedResults, docsRequested);
    }

//    public void run() {
//        normalize(docs1, docs2);
//        HashMap<String, Double> results = combine(docs1, docs2, alpha);
//        HashMap<String, Double> sortedResults = QueryResultsIO.sortByValue(results);
//        writer.writeLines(qid, sortedResults, docsRequested);
//    }

    private static void normalize(HashMap<String, Double> ...maps){
        for (HashMap<String, Double> docMap : maps) {
            double sum = 0;
            for (Double score : docMap.values()) {
//                if (!Double.isNaN(score))
                    sum += score;
            }
            for (Map.Entry<String, Double> pair : docMap.entrySet())
//                if (!Double.isNaN(pair.getValue()))
                    docMap.put(pair.getKey(), pair.getValue() / sum);
        }
    }

     private static HashMap<String, Double> combine(HashMap<String, Double> map1,
                                                    HashMap<String, Double> map2,
                                                    double alpha) {
        HashMap<String, Double> combinedMap = new HashMap<>();
        for (Map.Entry<String, Double> docPair : map1.entrySet()) {
             String docKey = docPair.getKey();
             if (map2.containsKey(docKey)) {
                 double combinedVal = (1 - alpha) * map1.get(docKey) + alpha * map2.get(docKey);
                 combinedMap.put(docKey, combinedVal);
             }
        }
        return combinedMap;
    }
}
