//
//  LLMBase.swift
//  Created by Guinmoon.

import Foundation
import llmfarm_core_cpp

public enum ModelLoadError: Error {
    // Throw when an invalid password is entered
    case modelLoadError

    // Throw when an expected resource is not found
    case contextLoadError

    case grammarLoadError

//        // Throw in all other cases
//        case unexpected(code: Int)
}


//func bridge(_ obj : T) -> UnsafeMutableRawPointer {
//    return UnsafeMutableRawPointer(Unmanaged.passUnretained(obj).toOpaque())
//}
//
//func bridge(_ ptr : UnsafeMutableRawPointer) -> T? {
//    return Unmanaged.fromOpaque(ptr).takeUnretainedValue()
//}

//func bridge<T : AnyObject>(obj : T) -> UnsafeRawPointer {
//    return UnsafeRawPointer(Unmanaged.passUnretained(obj).toOpaque())
//}
//
//func bridge<T : AnyObject>(ptr : UnsafeRawPointer) -> T {
//    return Unmanaged<T>.fromOpaque(ptr).takeUnretainedValue()
//}



public class LLMBase {

    public var context: OpaquePointer?
    public var grammar: OpaquePointer?
    public var contextParams: ModelAndContextParams
    public var sampleParams: ModelSampleParams = .default
    public var core_resourses = get_core_bundle_path()
    public var session_tokens: [Int32] = []
    public var modelLoadProgressCallback: ((Float)  -> (Bool))? = nil
    public var modelLoadCompleteCallback: ((String)  -> ())? = nil
    public var evalCallback: ((Int)  -> (Bool))? = nil
    public var evalDebugCallback: ((String)  -> (Bool))? = nil
    public var modelPath: String
    public var outputRepeatTokens: [ModelToken] = []

    // Used to keep old context until it needs to be rotated or purge out for new tokens
    var past: [[ModelToken]] = [] // Will house both queries and responses in order
    //var n_history: Int32 = 0
    var nPast: Int32 = 0

    var energyEvents: [String: String] = [:]
    var timingEvents: [llama_timings] = []
    var decodeCounter: Int = 0



    public  init(path: String, contextParams: ModelAndContextParams = .default) throws {

        energyEvents["init_model.start"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)

        self.modelPath = path
//        self.modelLoadProgressCallback = model_load_progress_callback
        self.contextParams = contextParams
        //        var params = gptneox_context_default_params()

        // Check if model file exists
        if !FileManager.default.fileExists(atPath: self.modelPath) {
            throw ModelError.modelNotFound(self.modelPath)
        }
        // load_model()

        energyEvents["init_model.end"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
    }

    public func load_model() throws {

        energyEvents["load_model.start"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)

         // Load model at path
        //        self.context = gptneox_init_from_file(path, params)
        //        let test = test_fn()
        var load_res:Bool? = false
        var params = gpt_context_default_params()
        params.n_ctx = contextParams.context
        params.n_parts = contextParams.parts
        params.seed = 0
        params.f16_kv = contextParams.f16Kv
        params.logits_all = contextParams.logitsAll
        params.vocab_only = contextParams.vocabOnly
        params.use_mlock = contextParams.useMlock
        params.embedding = contextParams.embedding
        do{
            try ExceptionCather.catchException {
                energyEvents["load_model.llm_load_model.start"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
                load_res = try? self.llm_load_model(path:self.modelPath,contextParams:contextParams,params: params)
                energyEvents["load_model.llm_load_model.end"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
            }

            if load_res != true{
                throw ModelLoadError.modelLoadError
            }

            print("%s: seed = %d\n", params.seed);

            if self.contextParams.grammar_path != nil && self.contextParams.grammar_path! != ""{
                try? self.load_grammar(self.contextParams.grammar_path!)
            }

            print(String(cString: print_system_info()))
            try ExceptionCather.catchException {
                _ = try? self.llm_init_logits()
            }
    //        if exception != nil{
    //            throw ModelError.failedToEval
    //        }

            print("Logits inited.")
        }catch {
            print(error)
            throw error
        }
        energyEvents["load_model.end"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
    }

    public func load_clip_model() -> Bool{
        return true
    }

    public func deinit_clip_model(){

    }

    public func destroy_objects(){

    }

    deinit {
        print("deinit LLMBase")
    }

    public func load_grammar(_ path:String) throws -> Void{
        do{
            try ExceptionCather.catchException {
                self.grammar = llama_load_grammar(path)
            }
        }
        catch {
            print(error)
            throw error
        }
    }

    public  func llm_load_model(path: String = "",
                                contextParams: ModelAndContextParams = .default,
                                params:gpt_context_params) throws -> Bool
    {
        return false
    }


    public func llm_token_nl() -> ModelToken{
        return 13
    }

    public func llm_token_bos() -> ModelToken{
        return gpt_base_token_bos()
    }

    public func llm_token_eos() -> ModelToken{
        return gpt_base_token_eos()
    }

    func llm_n_vocab(_ ctx: OpaquePointer!) -> Int32{
        return gpt_base_n_vocab(ctx)
    }

    func llm_get_logits(_ ctx: OpaquePointer!) -> UnsafeMutablePointer<Float>?{
        return gpt_base_get_logits(ctx);
    }

    func llm_get_n_ctx(ctx: OpaquePointer!) -> Int32{
        return gpt_base_n_ctx(ctx)
    }

    public func make_image_embed(_ image_path:String) -> Bool{
        return true
    }

    // Simple topK, topP, temp sampling, with repeat penalty
    func llm_sample() -> ModelToken {
        // Model input context size
        let ctx = self.context
        var last_n_tokens =  self.outputRepeatTokens
        let temp = self.sampleParams.temp
//        let top_k = self.sampleParams.top_k
        let top_p = self.sampleParams.top_p
        let tfs_z = self.sampleParams.tfs_z
        let typical_p = self.sampleParams.typical_p
//        let repeat_last_n = self.sampleParams.repeat_last_n
        let repeat_penalty = self.sampleParams.repeat_penalty
        let alpha_presence = self.sampleParams.presence_penalty
        let alpha_frequency = self.sampleParams.frequence_penalty
        let mirostat = self.sampleParams.mirostat
        let mirostat_tau = self.sampleParams.mirostat_tau
        let mirostat_eta = self.sampleParams.mirostat_eta
        let penalize_nl = self.sampleParams.penalize_nl

        let n_ctx = llm_get_n_ctx(ctx: ctx)
        // Auto params

        let top_k = self.sampleParams.top_k <= 0 ? llm_n_vocab(ctx) : self.sampleParams.top_k
        let repeat_last_n = self.sampleParams.repeat_last_n < 0 ? n_ctx : self.sampleParams.repeat_last_n

        //
        let vocabSize = llm_n_vocab(ctx)
        guard let logits = llm_get_logits(ctx) else {
            print("GPT sample error logits nil")
            return 0
        }
        var candidates = Array<llama_dadbed9_token_data>()
        for i in 0 ..< vocabSize {
            candidates.append(llama_dadbed9_token_data(id: i, logit: logits[Int(i)], p: 0.0))
        }
        var candidates_p = llama_dadbed9_token_data_array(data: candidates.mutPtr, size: candidates.count, sorted: false)

        // Apply penalties
        let nl_token = Int(llm_token_nl())
        let nl_logit = logits[nl_token]
        let last_n_repeat = min(min(Int32(last_n_tokens.count), repeat_last_n), n_ctx)

        llama_dadbed9_sample_repetition_penalty(&candidates_p,
                    last_n_tokens.mutPtr.advanced(by: last_n_tokens.count - Int(repeat_last_n)),
                    Int(repeat_last_n), repeat_penalty)
        llama_dadbed9_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                    last_n_tokens.mutPtr.advanced(by: last_n_tokens.count - Int(repeat_last_n)),
                    Int(last_n_repeat), alpha_frequency, alpha_presence)
        if(!penalize_nl) {
            logits[nl_token] = nl_logit
        }

        if (self.grammar != nil ) {
//            llama_sample_grammar(ctx,&candidates_p, self.grammar)
             llama_sample_grammar_for_dadbed9(ctx,&candidates_p, self.grammar)
        }

        var res_token:Int32 = 0

        if(temp <= 0) {
            // Greedy sampling
            res_token = llama_dadbed9_sample_token_greedy(ctx, &candidates_p)
        } else {
            var class_name = String(describing: self)
            if(mirostat == 1) {
                var mirostat_mu: Float = 2.0 * mirostat_tau
                let mirostat_m = 100
                llama_dadbed9_sample_temperature(ctx, &candidates_p, temp)
                if class_name != "llmfarm_core.LLaMa" && class_name != "llmfarm_core.LLaMa_MModal"{
                    res_token =  llama_dadbed9_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, Int32(mirostat_m), &mirostat_mu, vocabSize);
                }else{
                    res_token =  llama_sample_token_mirostat_for_dadbed9(ctx, &candidates_p, mirostat_tau, mirostat_eta, Int32(mirostat_m), &mirostat_mu);
                }
            } else if(mirostat == 2) {
                var mirostat_mu: Float = 2.0 * mirostat_tau
                llama_dadbed9_sample_temperature(ctx, &candidates_p, temp)
                if class_name != "llmfarm_core.LLaMa" && class_name != "llmfarm_core.LLaMa_MModal"{
                    res_token =  llama_dadbed9_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu)
                }
                else{
                    res_token =  llama_sample_token_mirostat_v2_for_dadbed9(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu)
                }
            } else {
                // Temperature sampling
                llama_dadbed9_sample_top_k(ctx, &candidates_p, top_k, 1)
                llama_dadbed9_sample_tail_free(ctx, &candidates_p, tfs_z, 1)
                llama_dadbed9_sample_typical(ctx, &candidates_p, typical_p, 1)
                llama_dadbed9_sample_top_p(ctx, &candidates_p, top_p, 1)
                llama_dadbed9_sample_temperature(ctx, &candidates_p, temp)
                if class_name != "llmfarm_core.LLaMa" && class_name != "llmfarm_core.LLaMa_MModal"{
                    res_token = llama_dadbed9_sample_token(ctx, &candidates_p)
                }else{
                    res_token = llama_sample_token_for_dadbed9(ctx, &candidates_p)
                }
            }
        }

        if (self.grammar != nil) {
            llama_grammar_accept_token(ctx, self.grammar, res_token);
        }
        return res_token

    }


    public func load_state(){}

    public func save_state(){}


    public func llm_eval(inputBatch: inout [ModelToken]) throws -> Bool{
        return false
    }

    public func llm_eval_clip() throws -> Bool{
        return true
    }

    func llm_init_logits() throws -> Bool {
        do{
            var inputs = [llm_token_bos(),llm_token_eos()]
            try ExceptionCather.catchException {
                _ = try? llm_eval(inputBatch: &inputs)
            }
            return true
        }
        catch{
            print(error)
            throw error
        }
    }


    public func llm_token_to_str(outputToken:Int32) -> String? {
        if let cStr = gpt_base_token_to_str(context, outputToken){
            return String(cString: cStr)
        }
        return nil
    }


    public func _eval_system_prompt(system_prompt:String? = nil) throws{
        if system_prompt != nil{
            var system_pormpt_Tokens = tokenizePrompt(system_prompt ?? "", .None)
            var eval_res:Bool? = nil
            try ExceptionCather.catchException {
                eval_res = try? self.llm_eval(inputBatch: &system_pormpt_Tokens)
            }
            if eval_res == false{
                throw ModelError.failedToEval
            }
            self.nPast += Int32(system_pormpt_Tokens.count)
        }
    }

    public func _eval_img(img_path:String? = nil) throws{
        if img_path != nil{
            do {
                try ExceptionCather.catchException {
                    _ = self.load_clip_model()
                    _ = self.make_image_embed(img_path!)
                    _ = try? self.llm_eval_clip()
                    self.deinit_clip_model()
                }
             }catch{
                print(error)
                throw error
             }
        }
    }

    public func kv_shift() throws{
        self.nPast = self.nPast / 2
        try ExceptionCather.catchException {
            var in_batch = [self.llm_token_eos()]
            _ = try? self.llm_eval(inputBatch: &in_batch)
        }
        print("Context Limit!")
    }

    public func chekc_skip_tokens (_ token:Int32) ->Bool{
        for skip in self.contextParams.skip_tokens{
            if skip == token{
                return false
            }
        }
        return true
    }

    public func eval_input_tokens_batched(inputTokens: inout [ModelToken],callback: ((String, Double) -> Bool), batchCounter: inout Int) throws -> Void {
        var inputBatch: [ModelToken] = []
        while inputTokens.count > 0 {
            inputBatch.removeAll()
            // See how many to eval (up to batch size??? or can we feed the entire input)
            // Move tokens to batch
            let evalCount = min(inputTokens.count, Int(sampleParams.n_batch))
            inputBatch.append(contentsOf: inputTokens[0 ..< evalCount])
            inputTokens.removeFirst(evalCount)

            energyEvents["generate.\(self.decodeCounter).decode.\(batchCounter).start"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
            if self.nPast + Int32(inputBatch.count) >= self.contextParams.context{
                try self.kv_shift()
                callback("**C_LIMIT**",0)
            }
            var eval_res:Bool? = nil
            try ExceptionCather.catchException {
                eval_res = try? self.llm_eval(inputBatch: &inputBatch)
            }
            if eval_res == false{
                throw ModelError.failedToEval
            }
            energyEvents["generate.\(self.decodeCounter).decode.\(batchCounter).end"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
            batchCounter += 1
            self.nPast += Int32(evalCount)
        }
    }

    public func predict(_ input: String, _ callback: ((String, Double) -> Bool),system_prompt:String? = nil,img_path: String? = nil ) throws -> String {
        llama_reset_timings(self.context)
        energyEvents["generate.\(self.decodeCounter).start"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
        energyEvents["prompt.\(self.decodeCounter).start"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
        //Eval system prompt then image if it's not nil
        try _eval_system_prompt(system_prompt:system_prompt)
        try _eval_img(img_path:img_path)

        let contextLength = Int32(contextParams.context)
        print("Past token count: \(nPast)/\(contextLength) (\(past.count))")
        // Tokenize with prompt format
        energyEvents["prompt.\(self.decodeCounter).tokenize.start"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
        var inputTokens = tokenizePrompt(input, self.contextParams.promptFormat)
        energyEvents["prompt.\(self.decodeCounter).tokenize.end"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
        if inputTokens.count == 0 && img_path == nil{
            return "Empty input."
        }
        let inputTokensCount = inputTokens.count
        print("Input tokens: \(inputTokens)")

        if inputTokensCount > contextLength {
            throw ModelError.inputTooLong
        }
        energyEvents["prompt.\(self.decodeCounter).end"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)

        var inputBatch: [ModelToken] = []
        var batchCounter = 0
        do {
            //Batched Eval all input tokens
            try eval_input_tokens_batched(inputTokens: &inputTokens,callback:callback, batchCounter: &batchCounter)
            // Output
            outputRepeatTokens = []
            var output = [String]()
            // Loop until target count is reached
            var completion_loop = true
            let eos_token = llm_token_eos()
            var llmSampleCounter = 0
            while completion_loop {
                // Pull a generation from context
                var outputToken:Int32 = -1
                try ExceptionCather.catchException {
                    energyEvents["generate.\(self.decodeCounter).llm_sample.\(llmSampleCounter).start"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
                    outputToken = self.llm_sample()
                    energyEvents["generate.\(self.decodeCounter).llm_sample.\(llmSampleCounter).end"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
                    llmSampleCounter += 1
                }
                // Repeat tokens update
                outputRepeatTokens.append(outputToken)
                if outputRepeatTokens.count > sampleParams.repeat_last_n {
                    outputRepeatTokens.removeFirst()
                }
                // Check for eos - end early - check eos before bos in case they are the same
                if outputToken == eos_token {
                    completion_loop = false
                    print("[EOS]")
                    break
                }
                // Check for BOS and tokens in skip list
                var skipCallback = false
                if !self.chekc_skip_tokens(outputToken){
                    print("Skip token: \(outputToken)")
                    skipCallback = true
                }
                energyEvents["generate.\(self.decodeCounter).token_to_string.start"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
                // Convert token to string and callback
                if !skipCallback, let str = llm_token_to_str(outputToken: outputToken){
                    output.append(str)
                    // Per token callback
                     let (output, time) = Utils.time {
                         return str
                     }
                     if callback(output, time) {
                        // Early exit if requested by callback
                        print(" * exit requested by callback *")
                        completion_loop = false
                        break
                    }
                }
                // Max output tokens count reached
                if (self.contextParams.n_predict != 0 && output.count>self.contextParams.n_predict){
                    print(" * n_predict reached *")
                    completion_loop = false
                    break
                }
                energyEvents["generate.\(self.decodeCounter).token_to_string.end"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)

                // Check if we need to run another response eval
                if completion_loop {
                    // Send generated token back into model for next generation
                    var eval_res:Bool? = nil
                    energyEvents["generate.\(self.decodeCounter).decode.\(batchCounter).start"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
                    if self.nPast >= self.contextParams.context - 2{
                        try self.kv_shift()
                        _ = callback("**C_LIMIT**",0)
                    }
                    try ExceptionCather.catchException {
                        inputBatch = [outputToken]
                        eval_res = try? self.llm_eval(inputBatch: &inputBatch)
                    }
                    if eval_res == false{
                        print("Eval res false")
                        throw ModelError.failedToEval
                    }
                    energyEvents["generate.\(self.decodeCounter).decode.\(batchCounter).end"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
                    batchCounter += 1
                    nPast += 1
                }
            }

            print("Total tokens: \(inputTokensCount + output.count) (\(inputTokensCount) -> \(output.count))")
            // print("Past token count: \(nPast)/\(contextLength) (\(past.count))")
            // Return full string for case without callback

            energyEvents["generate.\(self.decodeCounter).end"] = String(format: "%.0f", Date().timeIntervalSince1970 * 1_000_000_000)
            self.decodeCounter+=1

            let timings = llama_get_timings(self.context)
            timingEvents.append(timings)

            return output.joined()
        }catch{
            print(error)
            throw error
        }
    }

//     public func predict_ex(_ input: String, _ callback: ((String, Double) -> Bool),system_prompt:String? = nil,img_path: String? = nil ) throws -> String {
//         let params = sampleParams

//         try _eval_system_prompt(system_prompt:system_prompt)
//         try _eval_img(img_path:img_path)

//         let contextLength = Int32(contextParams.context)
//         print("Past token count: \(nPast)/\(contextLength) (\(past.count))")
//         // Tokenize with prompt format
//         var inputTokens = tokenizePrompt(input, self.contextParams.promptFormat)
//         if inputTokens.count == 0{
//             return "Empty input."
//         }
//         // self.session_tokens.append(contentsOf: inputTokens)
//         let inputTokensCount = inputTokens.count
//         print("Input tokens: \(inputTokens)")
//         // Add new input tokens to past array
//         past.append(inputTokens)
//         // Create space in context if needed
//         if inputTokensCount > contextLength {
//             throw ModelError.inputTooLong
//         }
// //        var totalLength = nPast + Int32(inputTokensCount)
//         // Input
//         var inputBatch: [ModelToken] = []
//         do {
//             while inputTokens.count > 0 {
//                 inputBatch.removeAll()
//                 // See how many to eval (up to batch size??? or can we feed the entire input)
//                 // Move tokens to batch
//                 let evalCount = min(inputTokens.count, Int(params.n_batch))
//                 inputBatch.append(contentsOf: inputTokens[0 ..< evalCount])
//                 inputTokens.removeFirst(evalCount)

//                 if self.nPast + Int32(inputBatch.count) >= self.contextParams.context{
//                     try self.kv_shift()
//                     callback("**C_LIMIT**",0)
//                 }
//                 var eval_res:Bool? = nil
//                 try ExceptionCather.catchException {
//                     eval_res = try? self.llm_eval(inputBatch: &inputBatch)
//                 }
//                 if eval_res == false{
//                     throw ModelError.failedToEval
//                 }
//                 nPast += Int32(evalCount)
//             }
//             // Output
//             outputRepeatTokens = []
//             var outputTokens: [ModelToken] = []
//             var output = [String]()
//             // Loop until target count is reached
//             var completion_loop = true
//             while completion_loop {
//                 // Pull a generation from context
//                 var outputToken:Int32 = -1
//                 try ExceptionCather.catchException {
//                     outputToken = self.llm_sample()
//                 }
//                 // Add output token to array
//                 outputTokens.append(outputToken)
//                 past.append([outputToken])
//                 // Repeat tokens update
//                 outputRepeatTokens.append(outputToken)
//                 if outputRepeatTokens.count > params.repeat_last_n {
//                     outputRepeatTokens.removeFirst()
//                 }
//                 // Check for eos - end early - check eos before bos in case they are the same
//                 if outputToken == llm_token_eos() {
//                     completion_loop = false
//                     print("[EOS]")
//                     break
//                 }
//                 // Check for bos, skip callback if so, bos = eos for most gptneox so this should typically never occur
//                 var skipCallback = false
//                 // if outputToken == llm_token_bos()  {
//                 //     print("[BOS]")
//                 //     skipCallback = true
//                 // }
//                 if !self.chekc_skip_tokens(outputToken){
//                     print("Skip token: \(outputToken)")
//                     skipCallback = true
//                 }
//                 // Convert token to string and callback
//                 // self.session_tokens.append(outputToken)
//                 if !skipCallback, let str = llm_token_to_str(outputToken: outputToken){
//                     output.append(str)
//                     // Per token callback
//                      let (output, time) = Utils.time {
//                          return str
//                      }
//                      if callback(output, time) {
//                         // Early exit if requested by callback
//                         print(" * exit requested by callback *")
//                         completion_loop = false //outputRemaining = 0
//                         break
//                     }
//                 }
//                 // Check if we need to run another response eval
//                 if completion_loop {
//                     // Send generated token back into model for next generation
//                     var eval_res:Bool? = nil
//                     if self.nPast >= self.contextParams.context - 2{
//                         try self.kv_shift()
//                         callback("**C_LIMIT**",0)
//                     }
//                     try ExceptionCather.catchException {
//                         var in_batch = [outputToken]
//                         eval_res = try? self.llm_eval(inputBatch: &in_batch)
//                     }
//                     if eval_res == false{
//                         print("Eval res false")
//                         throw ModelError.failedToEval
//                     }
//                     // Increment past count
//                     nPast += 1
//                 }
//             }
//             // Update past with most recent response
//             past.append(outputTokens)
//             print("Total tokens: \(inputTokensCount + outputTokens.count) (\(inputTokensCount) -> \(outputTokens.count))")
//             print("Past token count: \(nPast)/\(contextLength) (\(past.count))")
//             // Return full string for case without callback
//             return output.joined()
//         }catch{
//             print(error)
//             throw error
//         }
//     }

//    public func embeddings(_ input: String) throws -> [Float] {
//        // Tokenize the prompt
//        let inputs = llm_tokenize(input)
//
//        guard inputs.count > 0 else {
//            return []
//        }
//
//        _ = try llm_eval(inputBatch: inputs)
//
//        let embeddingsCount = Int(gpt_base_n_embd(context))
//        guard let embeddings = gpt_base_get_embeddings(context) else {
//            return []
//        }
//        return Array(UnsafeBufferPointer(start: embeddings, count: embeddingsCount))
//    }

    public func llm_tokenize(_ input: String, add_bos: Bool? = nil, parse_special:Bool? = nil) -> [ModelToken] {
        if input.count == 0 {
            return []
        }

        var embeddings = Array<ModelToken>(repeating: gpt_token(), count: input.utf8.count)
        let n = gpt_base_tokenize(context, input, &embeddings, Int32(input.utf8.count), self.contextParams.add_bos_token)
        if n<=0{
            return []
        }
        if Int(n) <= embeddings.count {
            embeddings.removeSubrange(Int(n)..<embeddings.count)
        }

        if self.contextParams.add_eos_token {
            embeddings.append(gpt_base_token_eos())
        }

        return embeddings
    }

    public func tokenizePrompt(_ input: String, _ style: ModelPromptStyle) -> [ModelToken] {
        switch style {
        case .None:
            return llm_tokenize(input)
        case .Custom:
            var formated_input = self.contextParams.custom_prompt_format.replacingOccurrences(of: "{{prompt}}", with: input)
            formated_input = formated_input.replacingOccurrences(of: "{prompt}", with: input)
            formated_input = formated_input.replacingOccurrences(of: "\\n", with: "\n")
            return llm_tokenize(formated_input)
         }
    }

    public func parse_skip_tokens(){
        // This function must be called after model loaded
        // Add BOS token to skip
        self.contextParams.skip_tokens.append(self.llm_token_bos())

        let splited_skip_tokens = self.contextParams.skip_tokens_str.components(separatedBy: [","])
        for word in splited_skip_tokens{
            let tokenized_skip = self.llm_tokenize(word,add_bos: false,parse_special: true)
            // Add only if tokenized text is one token
            if tokenized_skip.count == 1{
                self.contextParams.skip_tokens.append(tokenized_skip[0])
            }
        }

    }

    public func resetContext() {
        self.session_tokens.removeAll()
        self.past.removeAll()
        self.nPast = 0
    }

    public func resetEnergyEvents() {
        self.energyEvents.removeAll();
        self.decodeCounter = 0
    }

    public func resetTimingEvents() {
        self.timingEvents.removeAll();
    }

    public func saveTimingEventsToCSV(withFileName fileName: String) {

        var csvString = "t_start_ms,t_end_ms,t_load_ms,t_sample_ms,t_p_eval_ms,t_eval_ms,n_sample,n_p_eval,n_eval\n"
        for event in timingEvents {
            csvString += "\(event.t_start_ms),\(event.t_end_ms),\(event.t_load_ms),\(event.t_sample_ms),\(event.t_p_eval_ms),\(event.t_eval_ms),\(event.n_sample),\(event.n_p_eval),\(event.n_eval)\n"
        }

        // Writing to a file in the Document directory
        if let documentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = documentDirectory.appendingPathComponent("\(fileName).csv")

            do {
                try csvString.write(to: fileURL, atomically: true, encoding: .utf8)
                print("CSV file successfully saved at \(fileURL)")
            } catch {
                print("Error writing CSV file: \(error)")
            }
        }
    }

    public func saveEnergyEventsToCSV(withFileName fileName: String) {

        var csvString = ""
        for (key, value) in energyEvents {
            csvString += "\(key),\(value)\n"
        }

        // Writing to a file in the Document directory
        if let documentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = documentDirectory.appendingPathComponent("\(fileName).csv")

            do {
                try csvString.write(to: fileURL, atomically: true, encoding: .utf8)
                print("CSV file successfully saved at \(fileURL)")
            } catch {
                print("Error writing CSV file: \(error)")
            }
        }
    }
}


