import edu.stanford.nlp.math.ArrayMath;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.utility.Parameters;

import java.io.*;
import java.util.*;

public class WordEmbeddedScorer {

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
    private HashMap<String, HashMap<String, Double>> queryDocumentPair = new HashMap<>();
    private HashMap<String, String> queriesPair = new HashMap<>();
    private List<String> queries = new ArrayList<>();
    private HashMap<String, List<String>> documents = new HashMap<>();

    public WordEmbeddedScorer(Word2Vec dV, Word2Vec qV, HashMap<String, HashMap<String, Double>> queryDocumentNames) throws UnsupportedEncodingException, FileNotFoundException {
        if(documentVector == null){
            documentVector = dV;
            queryVector = qV;
        }
        for(String queryID : queryDocumentNames.keySet()){
            List<String> documentNames = new ArrayList<>(queryDocumentNames.get(queryID).keySet());
            documents.put(queryID, documentNames);
        }
        readParameters();
        computeDocumentQueryScores();
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

    private void computeDocumentQueryScores() throws UnsupportedEncodingException, FileNotFoundException {
        // open output file
        ResultWriter resultWriter = new ResultWriter("test-document-output.txt", false);

        for(int i = 0; i < queries.size(); i ++){
            HashMap<String, Double> documentScores = new HashMap<>();
            String queryNumber = queries.get(i);
            String[] queryTerms = queriesPair.get(queryNumber).split(" ");
            for(String document : documents.get(queryNumber)){
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
            resultWriter.write(queryNumber, sortByValue(documentScores));
            queryDocumentPair.put(queryNumber, sortByValue(documentScores));
            System.out.print(String.format("\r Processed all Documents for Query %s/%2d", i, queries.size()));
        }
        resultWriter.close();
    }

    public HashMap<String, HashMap<String, Double>> getFinalResults(){
        return queryDocumentPair;
    }

    public List<String> getQueries(){
        return queries;
    }

    public HashMap<String, Double> sortByValue(HashMap<String, Double> hm)
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
