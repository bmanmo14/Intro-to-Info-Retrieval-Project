import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.lemurproject.galago.core.eval.Eval;
import org.lemurproject.galago.core.parse.stem.KrovetzStemmer;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.prf.ExpansionModel;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.utility.Parameters;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
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

    public static void main(String[] args) throws Exception{
        String outputFileName = "word-embedding-model.txt";
        Word2Vec queryWordVectors = WordVectorSerializer.readWord2VecModel(new File(queryWordVectorPath));
        Word2Vec documentWordVectors = WordVectorSerializer.readWord2VecModel(new File(dHatVectorPath));
        new BatchSearch().retrieve(indexPath, outputFileName, queryWordVectors, documentWordVectors);
    }

    public static List<Parameters> readParameters(boolean rm){
        List<Parameters> queries = new ArrayList<Parameters>();
        try {
            File file = new File(queryPath);

            BufferedReader br = new BufferedReader(new FileReader(file));

            String st;
            while ((st = br.readLine()) != null) {
                String[] words = st.split("\t");
                if(rm)
                    queries.add(Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"#rm(%s)\"}", words[0], words[1]))));
                else
                    queries.add(Parameters.parseString((String.format("{\"number\":\"%s\", \"text\":\"%s\"}", words[0], words[1]))));
            }
        } catch(Exception e){
            System.out.println(e);
        }
        return queries;
    }

    public void retrieve(String indexPath, String outputFileName, Word2Vec queryVectors, Word2Vec documentVectors) throws Exception {
        int requested = 1000; // number of documents to retrieve
        boolean append = false;
        boolean queryExpansion = true;
        // open index
        Parameters queryParams = Parameters.create();
        queryParams.set("index", indexPath);
        queryParams.set("verbose", true);
        queryParams.set("defaultTextPart", "postings.krovetz");
        queryParams.set("stemmerClass", KrovetzStemmer.class.getName());
        queryParams.set("rmStemmer", "org.lemurproject.galago.core.parse.stem.KrovetzStemmer");
        queryParams.set("smoothMethod", "dirichlet");
        queryParams.set("scorer", "dirichlet");
        queryParams.set("fbDocs", 10);
        queryParams.set("fbTerm", 50);
        queryParams.set("fbOrigWeight", 0.5);

        Retrieval retrieval = RetrievalFactory.create(queryParams);
        
        // load queries
        List <Parameters> queries = readParameters(true);
        //List <Parameters> queries = new ArrayList <Parameters> ();
        //queries.add(Parameters.parseString(String.format("{\"number\":\"%s\", \"text\":\"%s\"}", "301", "International Organized Crime")));

        // open output file
        ResultWriter resultWriter = new ResultWriter(outputFileName, append);

        // for each query, run it, get the results, print in TREC format
        for (Parameters query : queries) {
            String queryNumber = query.getString("number");
            String queryText = query.getString("text");
            queryText = queryText.toLowerCase(); // option to fold query cases -- note that some parameters may require upper case
            
            logger.info("Processing query #" + queryNumber + ": " + queryText);
            
            query.set("requested", requested);

            Node root = StructuredQuery.parse(queryText);
            Node transformed = retrieval.transformQuery(root, query);
            
            // Query Expansion
            if (queryExpansion){
                // This query expansion technique can be replaced by other approaches.
                //ExpansionModel qe = new org.lemurproject.galago.core.retrieval.prf.RelevanceModel3(retrieval);
                ExpansionModel qe = new MixtureFeedbackModel(retrieval, queryVectors, documentVectors);

                try{
                    query.set("requested", requested);
                    query.set("smoothMethod", "dirichlet");
                    query.set("scorer", "dirichlet");
                    query.set("mu", 1000.0);
                    Node expandedQuery = qe.expand(root.clone(), query.clone());  
                    transformed = retrieval.transformQuery(expandedQuery, query);
                } catch (Exception ex){
                    ex.printStackTrace();
                }
            }
            
            // run query
            List<ScoredDocument> results = retrieval.executeQuery(transformed, query).scoredDocuments;
            
            // print results
            resultWriter.write(queryNumber, results);
        }
        resultWriter.close();
        Eval eval = new Eval();
        Parameters parameters = Parameters.create();
        parameters.set("details", true);
        parameters.set("metrics", "map");
        parameters.set("judgments", queryJudgement);
        parameters.set("runs", outputFileName);
        eval.run(parameters, new PrintStream("map-word-embedding-model.txt"));
    }
}
