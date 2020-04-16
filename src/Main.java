/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
import edu.stanford.nlp.math.ArrayMath;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.lemurproject.galago.core.eval.Eval;
import java.lang.Math;
import org.lemurproject.galago.core.eval.QuerySetJudgments;
import org.lemurproject.galago.core.index.disk.DiskIndex;
import org.lemurproject.galago.core.index.stats.FieldStatistics;
import org.lemurproject.galago.core.parse.Document;
import org.lemurproject.galago.core.parse.stem.KrovetzStemmer;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.ScoredDocument;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.utility.Parameters;
import org.javatuples.Pair;

import java.io.*;;

import java.util.*;

/**
 * Kyle Price
 * Brandon Mouser
 */
public class Main {
    public static final String indexPath = "index";
    public static final String queryPath = "PsgRobust/robust04.descs.tsv";
    public static final String queryJudgement = "PsgRobust/PsgRobust.qrels";
    public static final String wordvecPath = "/Users/brandonmouser/Downloads/GoogleNews-vectors-negative300.bin";
    public static final String queryWordVectorPath = "QUERY_WORDS.txt";
    public static final String dHatVectorPath = "D_HAT.txt";
    public static final String resultsPath = "batchResults";
    public static final String combinedResultsPath = "combinedResults";
    public static final String WEResultsPath = "WEResults";
    public static final String altResultsPath = "altResults";
    public static final int requested = 1000;

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

    public static Retrieval createRet(String stemmer, String stemmerClass, String textPart,
                                      String scorer, String model, int docs, int term, double weight, boolean mode){
        try {
            Parameters queryParams = Parameters.create();
            queryParams.set("index", indexPath);
            queryParams.set("verbose", true);
            queryParams.set("defaultTextPart", textPart);
            queryParams.set("stemmerClass", stemmerClass);
            queryParams.set("rmStemmer", "org.lemurproject.galago.core.parse.stem." + stemmer);
            queryParams.set("smoothMethod", scorer);
            queryParams.set("scorer", scorer);
            if(mode){
                queryParams.set("relevanceModel", model);
                queryParams.set("fbDocs", docs);
                queryParams.set("fbTerm", term);
                queryParams.set("fbOrigWeight", weight);
            }
            return RetrievalFactory.create(queryParams);
        } catch(Exception e){
            System.out.println(e);
        }
        return null;
    }
    public static void write_q() throws Exception {
        Word2Vec wordVectors = WordVectorSerializer.readBinaryModel(new File(wordvecPath), false, false);
        PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream(queryWordVectorPath, false)), true, "UTF-8");

        List<Parameters> queries = readParameters(false);
        for(Parameters p : queries) {
            String[] query_string = p.getAsString("text").split(" ");
            for (String term : query_string) {
                if (wordVectors.hasWord(term)) {
                    out.print(term + " ");
                    double[] vector = wordVectors.getWordVector(term);
                    for (double item : vector) {
                        out.print(item);
                        out.print(" ");
                    }
                    out.println();
                }
            }
        }
        System.out.println("Processed queries");
        out.close();

    }

    public static void write_d_hat() throws Exception {
        Word2Vec wordVectors = WordVectorSerializer.readBinaryModel(new File(wordvecPath), false, false);
        Retrieval ret = createRet("KrovetzStemmer", KrovetzStemmer.class.getName(), "postings.krovetz",
                "dirichlet", "org.lemurproject.galago.core.retrieval.prf.RelevanceModel3", 20, 100, 0.25, true);
        FieldStatistics fs = ret.getCollectionStatistics ("#lengths:part=lengths()");
        DiskIndex index = new DiskIndex(indexPath);
        Document.DocumentComponents dc = new Document.DocumentComponents(false, false, true);
        PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream(dHatVectorPath, false)), true, "UTF-8");

        for(int i = 0; i < fs.documentCount; i++) {
            String name = index.getName((long) i);
            List<String> terms = index.getDocument(name, dc).terms;
            double[] wordVector = new double[wordVectors.getWordVector("the").length];

            int count = 0;
            for(String term : terms){
                if(wordVectors.hasWord(term)) {
                    double[] vector = wordVectors.getWordVector(term);
                    double norm = ArrayMath.L2Norm(vector);
                    ArrayMath.divideInPlace(vector, norm);
                    wordVector = ArrayMath.pairwiseAdd(wordVector, vector);
                    count += 1;
                }
            }
            ArrayMath.divideInPlace(wordVector, (double) count);
            out.print(name + " ");
            for(double item : wordVector){
                out.print(item);
                out.print(" ");
            }
            out.println();
            System.out.print("\rProcessed document: " + String.valueOf(i) + "/" + String.valueOf(fs.documentCount));
        }
        out.close();
    }

    public static Parameters setParams(List<Pair> paramTuplesStr,
                                       List<Pair> paramTuplesBool,
                                       List<Pair> paramTuplesInt) {

        Parameters params = Parameters.create();
        for (Pair p : paramTuplesStr) {
            params.set(((String) p.getValue0()), ((String) p.getValue1()));
        }
        for (Pair p : paramTuplesBool) {
            params.set(((String) p.getValue0()), ((Boolean) p.getValue1()));
        }
        for (Pair p : paramTuplesInt) {
            params.set(((String) p.getValue0()), ((Integer) p.getValue1()));
        }
        return params;
    }

    public static List<String> getAllDocumentNames() throws Exception {
        List<String> documentNames = new ArrayList<String>();
        Retrieval ret = createRet("KrovetzStemmer", KrovetzStemmer.class.getName(), "postings.krovetz",
                "dirichlet", "org.lemurproject.galago.core.retrieval.prf.RelevanceModel3", 20, 100, 0.25, true);
        FieldStatistics fs = ret.getCollectionStatistics ("#lengths:part=lengths()");
        DiskIndex index = new DiskIndex(indexPath);
        for(int i = 0; i < fs.documentCount; i++) {
            documentNames.add(index.getName((long) i));
        }
        return documentNames;
    }

    private static void evalAndCompare(String ... resultFiles) throws Exception {
        for (String resultFile: resultFiles) {
             Parameters evalParams = setParams(new ArrayList<>(Arrays.asList(Pair.with("metrics", "map"),
                                                                                 Pair.with("judgments", queryJudgement),
                                                                                 Pair.with("baseline", resultFile))),
                                               new ArrayList<>(Arrays.asList(Pair.with("details", true))),
                                               new ArrayList<>());


            QuerySetJudgments qsj = new QuerySetJudgments(queryJudgement);
            Parameters resultParams = Eval.singleEvaluation(evalParams, qsj, new ArrayList<>());

            Double map = ((Double) (resultParams.getMap("all").get("map")));
            System.out.println(String.format("MAP = %s, NDCG@10 = %s ", map.toString(), "X"));
        }
    }

    private static void waitOn(Thread ... threads) {
        try {
            for (Thread t : threads)
                t.join();
        }
        catch(java.lang.InterruptedException e) {
            System.out.println("an error occured while joining threads");
        }
    }

    public static HashMap<String, Double> execQuery(Parameters query, Retrieval ret) {
        HashMap<String, Double> resultDocuments = new HashMap<>();
        try {
            String queryNumber = query.getString("number");
            String queryText = query.getString("text");
            Node root = StructuredQuery.parse(queryText);
            Node transformed = ret.transformQuery(root, query);

            // run query
            List<ScoredDocument> results = ret.executeQuery(transformed, query).scoredDocuments;
            QueryResultsIO.resultDocuments(queryNumber, results, resultDocuments);

        } catch (java.lang.Exception e) {
            System.out.println("Error in OtherModelScorer Thread!!!");
        }
        return resultDocuments;
    }



    public static void main(String[] args) throws Exception {
        //write_d_hat();
        //write_q();

        double alpha = 0.5;
        String otherModel = "bm25";
//        String otherModel = "jm";

        List<String> documents = getAllDocumentNames();
        Parameters queryParams = setParams(new ArrayList<Pair>(Arrays.asList(Pair.with("queryFormat", "tsv"),
                                                                             Pair.with("query", queryPath),
                                                                             Pair.with("defaultTextPart", "postings.krovetz"),
                                                                             Pair.with("index", indexPath),
                                                                             Pair.with("scorer", otherModel))),
                                           new ArrayList<Pair>(),
                                           new ArrayList<Pair>(Arrays.asList(Pair.with("requested", documents.size()),
                                                                             Pair.with("mu", 1000)
//                                                                             Pair.with("lambda", 1)
                                                                             )));

        Word2Vec queryWordVectors = WordVectorSerializer.readWord2VecModel(new File(queryWordVectorPath));
        Word2Vec documentWordVectors = WordVectorSerializer.readWord2VecModel(new File(dHatVectorPath));

        List<Parameters> queries = QueryResultsIO.readParameters(queryPath, otherModel);
        Retrieval ret = RetrievalFactory.create(queryParams);
        QueryResultsIO writer = new QueryResultsIO();
        writer.newWriter(combinedResultsPath);
        System.out.println("Done reading vector files");

        Iterator it = queries.iterator();
        while (it.hasNext()) {
            // time it
            long start = System.nanoTime();

            Parameters q1 = (Parameters)it.next();
            HashMap<String, Double> embeddedResults1 = new HashMap<>();
            HashMap<String, Double> embeddedResults2 = new HashMap<>();
            HashMap<String, Double> embeddedResults3 = new HashMap<>();
            HashMap<String, Double> embeddedResults4 = new HashMap<>();

            WordEmbeddedScorer wordEmbeddedScorer1 = new WordEmbeddedScorer(embeddedResults1,
                                                                           documentWordVectors,
                                                                           queryWordVectors,
                                                                           documents,
                                                                           QueryResultsIO.createQueryPair(queryPath),
                                                                           q1.getAsString("number"));
            if (!it.hasNext()) {
                waitOn(wordEmbeddedScorer1);
                ModelMixer.mix( execQuery(q1, ret), embeddedResults1, writer, q1.getAsString("number"), alpha, requested);
                break;
            }
            Parameters q2 = (Parameters)it.next();
            WordEmbeddedScorer wordEmbeddedScorer2 = new WordEmbeddedScorer(embeddedResults2,
                                                                           documentWordVectors,
                                                                           queryWordVectors,
                                                                           documents,
                                                                           QueryResultsIO.createQueryPair(queryPath),
                                                                           q2.getAsString("number"));
            if (!it.hasNext()) {
                waitOn(wordEmbeddedScorer1, wordEmbeddedScorer2);
                ModelMixer.mix( execQuery(q1, ret), embeddedResults1, writer, q1.getAsString("number"), alpha, requested);
                ModelMixer.mix( execQuery(q2, ret), embeddedResults2, writer, q2.getAsString("number"), alpha, requested);
                break;
            }
            Parameters q3 = (Parameters)it.next();
            WordEmbeddedScorer wordEmbeddedScorer3 = new WordEmbeddedScorer(embeddedResults3,
                                                                           documentWordVectors,
                                                                           queryWordVectors,
                                                                           documents,
                                                                           QueryResultsIO.createQueryPair(queryPath),
                                                                           q3.getAsString("number"));

            if (!it.hasNext()) {
                waitOn(wordEmbeddedScorer1, wordEmbeddedScorer2, wordEmbeddedScorer3);
                ModelMixer.mix( execQuery(q1, ret), embeddedResults1, writer, q1.getAsString("number"), alpha, requested);
                ModelMixer.mix( execQuery(q2, ret), embeddedResults2, writer, q2.getAsString("number"), alpha, requested);
                ModelMixer.mix( execQuery(q3, ret), embeddedResults3, writer, q3.getAsString("number"), alpha, requested);
                break;
            }
            Parameters q4 = (Parameters)it.next();
            WordEmbeddedScorer wordEmbeddedScorer4 = new WordEmbeddedScorer(embeddedResults4,
                                                                            documentWordVectors,
                                                                            queryWordVectors,
                                                                            documents,
                                                                            QueryResultsIO.createQueryPair(queryPath),
                                                                            q4.getAsString("number"));

            // wait on all
            waitOn(wordEmbeddedScorer1, wordEmbeddedScorer2, wordEmbeddedScorer3, wordEmbeddedScorer4);

            // linearly combine
            ModelMixer.mix(execQuery(q1, ret), embeddedResults1, writer, q1.getAsString("number"), alpha, requested);
            ModelMixer.mix(execQuery(q2, ret), embeddedResults2, writer, q2.getAsString("number"), alpha, requested);
            ModelMixer.mix(execQuery(q3, ret), embeddedResults3, writer, q3.getAsString("number"), alpha, requested);
            ModelMixer.mix(execQuery(q4, ret), embeddedResults4, writer, q4.getAsString("number"), alpha, requested);

            long end = System.nanoTime();
            long duration = (end - start);

            System.out.print(String.format("\rProcessed queries: %s, %s, %s %s in %.0f seconds",
                                           q1.getAsString("number"),
                                           q2.getAsString("number"),
                                           q3.getAsString("number"),
                                           q4.getAsString("number"),
                                           duration / Math.pow(10, 9)));
        }
        evalAndCompare(combinedResultsPath);
    }
}
