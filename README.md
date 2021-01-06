# benchmark-flair-inference

Tested `ner-fast`:

Input size: 670k characters
Average article size: 2500 characters
Language: english

### CPU performance

CPU: 8, 1.6 GHz Intel Core i5
Total time: 6 mins
Time per 1k chars:
Articles per seconds:

### GPU performance

GPU: Tesla V100
Card memory: 16 GB
Accelerator: CUDA
Total time: 20 seconds
Time per 1k chars: 0.03 seconds
Articles per seconds: 13.3

#### Conclusion

Running on GPU offers ~20x performance boost comparing with CPU performance. I suppose, that it can also be run in parallel threads, because GPU has enough memory
to handle multiple models. I assume at least 4 models could be run in parallel, therefore the performance will be ~50 articles per second. Which is more than
needed for our use case.
