import org.lemurproject.galago.core.eval.Eval;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.utility.Parameters;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static org.lemurproject.galago.core.tools.apps.BatchSearch.logger;


public class BatchSearch {
    public static final int requested = 1000;
    public static final String indexPath = "index";
    public static final String queryPath = "PsgRobust/robust04.titles.tsv";
    public static final String queryJudgement = "PsgRobust/robust04.qrels";
    public static final String wordvecPath = "GoogleNews-vectors-negative300.bin";
    public static final String queryWordVectorPath = "QUERY_WORDS.txt";
    public static final String dHatVectorPath = "D_HAT.txt";

    public BatchSearch() throws Exception{
    }

    public static List<Parameters> readParameters(String type){
        List<Parameters> queries = new ArrayList<Parameters>();
        try {
            File file = new File(queryPath);

            BufferedReader br = new BufferedReader(new FileReader(file));

            String st;
            while ((st = br.readLine()) != null) {
                String[] words = st.split("\t");
                switch (type){
                    case "bm25":
                        queries.add(Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"#bm25(%s)\"}", words[0], words[1]))));
                        break;
                    case "rm":
                        queries.add(Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"#rm(%s)\"}", words[0], words[1]))));
                        break;
                    case "jm":
                        queries.add(Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"#jm(%s)\"}", words[0], words[1]))));
                        break;
                    default:
                        queries.add(Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"%s\"}", words[0], words[1]))));
                        break;
                }
            }
        } catch(Exception e){
            System.out.println(e);
        }
        return queries;
    }

    public HashMap<String, HashMap<String, Double>> retrieve(String outputFileName, Parameters queryParams, String type) throws Exception {
        int requested = 1000; // number of documents to retrieve
        boolean append = false;

        Retrieval retrieval = RetrievalFactory.create(queryParams);
        
        // load queries
        List <Parameters> queries = readParameters(type);

        // open output file
        ResultWriter resultWriter = new ResultWriter(outputFileName, append);

        HashMap<String, HashMap<String, Double>> resultDocuments = new HashMap<>();
        // for each query, run it, get the results, print in TREC format
        for (Parameters query : queries) {
            String queryNumber = query.getString("number");
            String queryText = query.getString("text");
            queryText = queryText.toLowerCase(); // option to fold query cases -- note that some parameters may require upper case
            
            logger.info("Processing query #" + queryNumber + ": " + queryText);
            
            query.set("requested", requested);

            Node root = StructuredQuery.parse(queryText);
            Node transformed = retrieval.transformQuery(root, query);

            // run query
            List<ScoredDocument> results = retrieval.executeQuery(transformed, query).scoredDocuments;
            
            // print results

            resultDocuments.put(queryNumber, resultWriter.write(queryNumber, results));
        }

        return resultDocuments;
    }
}
