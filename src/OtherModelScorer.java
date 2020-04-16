import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.utility.Parameters;

import java.util.HashMap;
import java.util.List;

public class OtherModelScorer extends Thread {
    private Parameters params;
    private Parameters query;
    private Retrieval ret;
    public HashMap<String, Double> resultDocuments;

    OtherModelScorer(HashMap<String, Double> resultDocuments, Retrieval ret, Parameters query, Parameters params) {
        this.resultDocuments = resultDocuments;
        this.params = params;
        this.query = query;
        this.ret = ret;
        this.start();
    }

//    public HashMap<String, Double> getFinalResults() {
//        return resultDocuments;
//    }

    public void run() {
        try {

            String queryNumber = query.getString("number");
            String queryText = query.getString("text");

            Node root = StructuredQuery.parse(queryText);
            Node transformed = ret.transformQuery(root, query);

            // run query
            List<ScoredDocument> results = ret.executeQuery(transformed, query).scoredDocuments;
            QueryResultsIO.resultDocuments(queryNumber, results, this.resultDocuments);
        } catch (java.lang.Exception e) {
            System.out.println("Error in OtherModelScorer Thread!!!");
        }
    }
}
