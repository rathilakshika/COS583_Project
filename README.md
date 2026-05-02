When Peter Shor wrote his seminal paper "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer" in 1994, the algorithm that he 
came up with was only the what we now call "logical" circuit (this term didn't exist until Shor himself discovered/invented quantum error correction a year
later) . It was not compiled down to the native gateset of a particular device, and it was completely unprotected against any errors that would occur. 
This project follows the path of taking the circuit for Shor's algorithm, encoding it into an error correcting code (the Steane code in our case), applying
logical operations, and then compiling it down to the native gateset of a real quantum device that has the adequate constraints to perform the operations of the circuit. 
Even though we chose a number as small as 15 to factor using the algorithm, the circuit still remained far too complicated to simulate on a classical computer once it was 
transformed into a compiled and error corrected circuit. However, we still wanted to also show how the simulation of an error corrected Shor's algorithm would look for 
N=15 and a=11. As such, we perform a precompilation trick, stemming from the fact that we do in fact know the period in this case. This way, we can simulate an equivalent,
and much simpler circuit on Stim. It is error corrected, decoded in time, and the results show the correct factors at the end of the notebook. 


There were multiple papers that we utilized directly in order to help guide us with this project. 

1) We determined that we need 2n+2 qubits for a space optimized run of Shor's algorithm from 
Takahashi and N. Kunihiro, "A quantum circuit for Shor's factoring algorithm using 2n+2 qubits," Quantum Information and Computation, vol. 6, no. 2, pp. 184–192, 2006.

2) We determined what the magic state factory for a Steane code should look like from T. Itogawa, Y. Takada, Y. Hirano, and K. Fujii, "Efficient magic state distillation by zero-level distillation," arXiv:2403.03991v2 [quant-ph], Jun. 2025.

3) We determined what the simplified circuit for N=15 and a=11 should look like from J. A. Smolin, G. Smith, and A. Vargo, "Oversimplifying quantum factoring," Nature, vol. 499, pp. 163–165, Jul. 2013. DOI: 10.1038/nature12290. Preprint: arXiv:1301.7007 [quant-ph]. 
and R. Rines, B. Hall, M. H. Teo, J. Viszlai, D. C. Cole, D. Mason, C. Barker, et al., "Demonstration of a logical architecture uniting motion and in-place entanglement: Shor's algorithm, constant-depth CNOT ladder, and many-hypercube code," arXiv:2509.13247 [quant-ph], Sep. 2025. 



Ultimately, throughout the initial compiled circuit, we remained committed to not "cheating" the circuit by using any tricks that arise solely from actually 
knowing the factors or the period. This manifested itself in a pretty complicated circuit that accomplishes a fairly trivial task, showcasing just how much work
overhead goes into encoding a universal set of gates into a simple error correcting code. It was quite a humbling and fun process, and taught us a lot about why so much effort is
being put precisely into reducing this overhead to make Shor's algorithm work. 
