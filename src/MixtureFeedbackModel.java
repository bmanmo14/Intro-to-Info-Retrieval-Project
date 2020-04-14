import edu.stanford.nlp.math.ArrayMath;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.lemurproject.galago.core.parse.Document;
import org.lemurproject.galago.core.parse.stem.Stemmer;
import org.lemurproject.galago.core.retrieval.Results;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.prf.ExpansionModel;
import org.lemurproject.galago.core.retrieval.prf.WeightedTerm;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.core.util.WordLists;
import org.lemurproject.galago.utility.Parameters;

import java.io.IOException;
import java.util.*;


public class MixtureFeedbackModel implements ExpansionModel{
    protected Retrieval retrieval;
    int defaultFbDocs, defaultFbTerms;
    double defaultFbOrigWeight;
    Set <String> exclusionTerms;
    Stemmer stemmer;
    static Word2Vec queryVector;
    static Word2Vec documentVector;
    
    public MixtureFeedbackModel (Retrieval r, Word2Vec qV, Word2Vec dV) throws IOException{
        retrieval = r;
        defaultFbDocs = (int) Math.round(r.getGlobalParameters().get("fbDocs", 10.0));
        defaultFbTerms = (int) Math.round(r.getGlobalParameters().get("fbTerm", 100.0));
        defaultFbOrigWeight = r.getGlobalParameters().get("fbOrigWeight", 0.2);
        exclusionTerms = WordLists.getWordList(r.getGlobalParameters().get("rmstopwords", "rmstop"));
        Parameters gblParms = r.getGlobalParameters();
        this.stemmer = FeedbackData.getStemmer(gblParms, retrieval);
        queryVector = qV;
        documentVector = dV;
    }
    
    public List<ScoredDocument> collectInitialResults(Node transformed, Parameters fbParams) throws Exception {
        Results results = retrieval.executeQuery(transformed, fbParams);
        List<ScoredDocument> res = results.scoredDocuments;
        if (res.isEmpty())
            throw new Exception("No feedback documents found!");
        return res;
    }
    
    public Node generateExpansionQuery(List<WeightedTerm> weightedTerms, int fbTerms) throws IOException, Exception {
        Node expNode = new Node("combine");
//        System.err.println("Feedback Terms:");
        for (int i = 0; i < Math.min(weightedTerms.size(), fbTerms); i++) {
          Node expChild = new Node("text", weightedTerms.get(i).getTerm());
          expNode.addChild(expChild);
          expNode.getNodeParameters().set("" + i, weightedTerms.get(i).getWeight());
        }
        return expNode;
    }
    
    public int getFbDocCount (Node root, Parameters queryParameters) throws Exception{
        int fbDocs = (int)Math.round(root.getNodeParameters().get("fbDocs", queryParameters.get("fbDocs", (double) defaultFbDocs)));
        if (fbDocs <= 0)
            throw new Exception ("Invalid number of feedback documents!");
        return fbDocs;
    }
    
    public int getFbTermCount (Node root, Parameters queryParameters) throws Exception{
        int fbTerms = (int) Math.round(root.getNodeParameters().get("fbTerm", queryParameters.get("fbTerm", (double) defaultFbTerms)));
        if (fbTerms <= 0)
            throw new Exception ("Invalid number of feedback terms!");
        return fbTerms;
    }
    
    public Node interpolate (Node root, Node expandedQuery, Parameters queryParameters) throws Exception{
        queryParameters.set("defaultFbOrigWeight", defaultFbOrigWeight);
        queryParameters.set("fbOrigWeight", queryParameters.get("fbOrigWeight", defaultFbOrigWeight));
        return linearInterpolation(root, expandedQuery, queryParameters);
    }
    
    public Node linearInterpolation (Node root, Node expNode, Parameters parameters) throws Exception{
        double defaultFbOrigWeight = parameters.get("defaultFbOrigWeight", -1.0);
        if (defaultFbOrigWeight < 0)
            throw new Exception ("There is not defaultFbOrigWeight parameter value");
        double fbOrigWeight = parameters.get("fbOrigWeight", defaultFbOrigWeight);
        if (fbOrigWeight == 1.0) {
            return root;
        }
        Node result = new Node("combine");
        result.addChild(root);
        result.addChild(expNode);
        result.getNodeParameters().set("0", fbOrigWeight);
        result.getNodeParameters().set("1", 1.0 - fbOrigWeight);
        return result;
    }
    
    public Parameters getFbParameters (Node root, Parameters queryParameters) throws Exception{
        Parameters fbParams = Parameters.create();
        fbParams.set("requested", getFbDocCount(root, queryParameters));
        fbParams.set("passageQuery", false);
        fbParams.set("extentQuery", false);
        fbParams.setBackoff(queryParameters);
        return fbParams;
    }

    @Override
    public Node expand(Node root, Parameters queryParameters) throws Exception {
        int fbTerms = getFbTermCount(root, queryParameters); 
        // transform query to ensure it will run
        Parameters fbParams = getFbParameters(root, queryParameters);
        Node transformed = retrieval.transformQuery(root.clone(), fbParams);
        
        // get some initial results
        List<ScoredDocument> initialResults = collectInitialResults(transformed, fbParams);
   
        
        // extract grams from results
        Set<String> queryTerms = getTerms(stemmer, StructuredQuery.findQueryTerms(transformed));
        FeedbackData feedbackData = new FeedbackData(retrieval, exclusionTerms, initialResults, fbParams);

        List <WeightedTerm> weightedTerms = computeWeights(feedbackData, fbParams, queryParameters, queryTerms);
        Collections.sort(weightedTerms);
        Node expNode = generateExpansionQuery(weightedTerms, fbTerms);
        
        return interpolate(root, expNode, queryParameters);
    }
    
    public static Set<String> getTerms(Stemmer stemmer, Set<String> terms) {
      if (stemmer == null)
          return terms;

      Set<String> stems = new HashSet<String>(terms.size());
      for (String t : terms) {
        String s = stemmer.stem(t);
        stems.add(s);
      }
      return stems;
    }

    public static List<WeightedTerm> scoreGrams(Map<String, Map<ScoredDocument, Integer>> counts,
                                                Map<String, Double> corpusProb, Map<ScoredDocument, Integer> doclength) throws IOException {
        List<WeightedTerm> grams = new ArrayList<WeightedTerm>();
        List<WeightedTerm> newGrams = new ArrayList<WeightedTerm>();
        Map<ScoredDocument, Integer> termCounts;
        float totalCount = 0;
        for (String term : counts.keySet()) {
            WeightedUnigram g = new WeightedUnigram(term);
            termCounts = counts.get(term);
            if(!queryVector.hasWord(term))
                continue;
            Double termCorpus = corpusProb.get(term);
            double[] termVector = queryVector.getWordVector(term);
            float termCount = 0;

            for (ScoredDocument sd : termCounts.keySet()) {
                double[] document = documentVector.getWordVector(sd.documentName);
                double top = ArrayMath.dotProduct(termVector, document);
                double bottom = ArrayMath.L2Norm(termVector) * ArrayMath.L2Norm(document);
                int length = doclength.get(sd);
                g.score += ((float)top / (float)bottom) * (((float)termCounts.get(sd) / (float)length));
                termCount += termCounts.get(sd);
                //g.score += scores.get(sd) * termCounts.get(sd) / length;
            }
            // 1 / fbDocs from the RelevanceModel source code
            //g.score *= (1.0 / scores.size());
            grams.add(g);
        }

        return grams;
        }

//        for (WeightedTerm g : grams) {
//            g.score = g.score / totalCount;
//            newGrams.add(g);
//        }
//
//        return newGrams;
//    }


    public List <WeightedTerm> computeWeights (FeedbackData feedbackData, Parameters fbParam, Parameters queryParameters, Set <String> queryTerms) throws Exception{

        long documentCount = (feedbackData.retrieval.getCollectionStatistics ("#lengths:part=lengths()")).collectionLength;

        Map<String, Double> corpusTermCount = new HashMap<String, Double>();
        Map<ScoredDocument, Integer> doclength = new HashMap<ScoredDocument, Integer>();
        Document.DocumentComponents corpusParams = new Document.DocumentComponents(false, false, true);

//        for(String query : feedbackData.termCounts.keySet()){
//            Node node = StructuredQuery.parse(query);
//            node.getNodeParameters().set("queryType", "count");
//            node = feedbackData.retrieval.transformQuery(node, Parameters.create());
//            double thing = (double)feedbackData.retrieval.getNodeStatistics(node).nodeFrequency/documentCount;
//            corpusTermCount.put(query, thing);
//        }

        for(ScoredDocument sd : feedbackData.initialResults){
            Document doc = retrieval.getDocument(sd.documentName, corpusParams);
            doclength.put(sd, doc.terms.size());
            for(String query : feedbackData.termCounts.keySet()) {
                //corpusTermCount.put(query, thing);
            }
        }

        // compute term weights
        List<WeightedTerm> scored = scoreGrams(feedbackData.termCounts, corpusTermCount, doclength);
        return scored;
    }
}
