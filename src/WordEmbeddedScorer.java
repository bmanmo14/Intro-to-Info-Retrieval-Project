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

//    public static final String indexPath = "index";
//    public static final String queryPath = "PsgRobust/PsgRobust.descs.tsv";
//    public static final String queryJudgement = "PsgRobust/robust04.qrels";
//    public static final String wordvecPath = "GoogleNews-vectors-negative300.bin";
//    public static final String queryWordVectorPath = "QUERY_WORDS.txt";
//    public static final String dHatVectorPath = "D_HAT.txt";
//    public static final String resultsPath = "batchResults";
//    public static final int requested = 1000;

    private static Word2Vec documentVector;
    private static Word2Vec queryVector;
    public HashMap<String, Double> queryDocumentPair;
    private HashMap<String, String> queryPair = new HashMap<>();
    private List<String> queries = new ArrayList<>();
    private List<String> documents = new ArrayList<>();
    private String queryID;
    private HashMap<String, Double> otherResults;

    public WordEmbeddedScorer(HashMap<String, Double> queryDocumentPair,
                              Word2Vec dV,
                              Word2Vec qV,
                              List<String> queryDocumentNames,
                              HashMap<String, String> qPair,
                              String qID,
                              HashMap<String, Double> otherResults)  {
        if(documentVector == null){
            documentVector = dV;
            queryVector = qV;
        }
        this.queryDocumentPair = queryDocumentPair;
        this.otherResults = otherResults;
        queryPair = qPair;
        documents = queryDocumentNames;
        queryID = qID;
        start();
    }

    public void run()  {

        String[] queryTerms = queryPair.get(queryID).split(" ");

        for(String document : documents) {
            if (!this.otherResults.containsKey(document))
                continue;
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
            if(!Double.isNaN(score))
                this.queryDocumentPair.put(document, score);
        }
    }
}
