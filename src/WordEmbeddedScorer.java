import edu.stanford.nlp.math.ArrayMath;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.utility.Parameters;

import java.io.*;
import java.lang.reflect.Parameter;
import java.util.*;

public class WordEmbeddedScorer extends Thread {

    public static final String indexPath = "index";
    public static final String queryPath = "PsgRobust/robust04.titles.tsv";
    public static final String queryJudgement = "PsgRobust/robust04.qrels";
    public static final String wordvecPath = "GoogleNews-vectors-negative300.bin";
    public static final String queryWordVectorPath = "QUERY_WORDS.txt";
    public static final String dHatVectorPath = "D_HAT.txt";
    public static final String resultsPath = "batchResults";
    public static final int requested = 1000;

    private static Word2Vec documentVector;
    private static Word2Vec queryVector;
    private HashMap<String, Double> queryDocumentPair = new HashMap<>();
    private HashMap<String, String> queriesPair = new HashMap<>();
    private List<String> queries = new ArrayList<>();
    private List<String> documents = new ArrayList<>();
    private String queryID;

    public WordEmbeddedScorer(Word2Vec dV, Word2Vec qV, List<String> queryDocumentNames) throws UnsupportedEncodingException, FileNotFoundException {
        if(documentVector == null){
            documentVector = dV;
            queryVector = qV;
        }
//        for(String queryID : queryDocumentNames.keySet()){
//            List<String> documentNames = new ArrayList<>(queryDocumentNames.get(queryID).keySet());
//            documents.put(queryID, documentNames);
//        }
        documents = queryDocumentNames;
        readParameters();
    }

    public void readParameters() {
        try {
            File file = new File(queryPath);
            BufferedReader br = new BufferedReader(new FileReader(file));
            String st;
            while ((st = br.readLine()) != null) {
                String[] words = st.split("\t");
                queriesPair.put(words[0], words[1]);
                queries.add(words[0]);
            }
        } catch(Exception e){
            System.out.println(e);
        }
    }

    public void startThread(String qID){
        queryID = qID;
        start();
    }

    public void run()  {
        HashMap<String, Double> documentScores = new HashMap<>();
        String[] queryTerms = queriesPair.get(queryID).split(" ");
        for(String document : documents){
            double[] documentVec = documentVector.getWordVector(document);
            double score = 0.0;
            int term_count = 0;
            for(String term : queryTerms){
                if(queryVector.hasWord(term)) {
                    double[] queryTerm = queryVector.getWordVector(term);
                    double top = ArrayMath.dotProduct(queryTerm, documentVec);
                    double bottom = ArrayMath.L2Norm(queryTerm) * ArrayMath.L2Norm(documentVec);
                    score += ((float) top / (float) bottom);
                    term_count += 1;
                }
            }
            score = score * (1/(double) term_count);
            documentScores.put(document, score);
        }
        queryDocumentPair = sortByValue(documentScores);
    }

    public HashMap<String, Double> getFinalResults(){
        return queryDocumentPair;
    }

    public List<String> getQueries(){
        return queries;
    }

    public static HashMap<String, Double> sortByValue(HashMap<String, Double> hm)
    {
        // Create a list from elements of HashMap
        List<Map.Entry<String, Double> > list =
                new LinkedList<Map.Entry<String, Double> >(hm.entrySet());

        // Sort the list
        Collections.sort(list, new Comparator<Map.Entry<String, Double> >() {
            public int compare(Map.Entry<String, Double> o1,
                               Map.Entry<String, Double> o2)
            {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });

        // put data from sorted list to hashmap
        HashMap<String, Double> temp = new LinkedHashMap<String, Double>();
        for (Map.Entry<String, Double> aa : list) {
            temp.put(aa.getKey(), aa.getValue());
        }
        return temp;
    }

}
