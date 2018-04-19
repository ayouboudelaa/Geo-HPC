/** 
	TP Géosciences HPC Master 1 - USTHB.
	Pour compiler: mpicc -o MM MM.c
	Pour executer: mpiexec -np 5 ./MM
	*
	*
	np: le nombre de processus.
	*
	**
	***
	@ayouboudelaa
**/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define MAX_RAND 10
#define N 15

#define foreach(a, b, c) for (int a = b; a < c; a++)

typedef float matrix[N][N];

// Remplir une matrice avec des nombres réeles aléatoires.
#define _fill(M) fillMatrix(M)
float * fillMatrix(matrix M) {
	foreach(i, 0, N) {
		foreach(j, 0, N) {
			M[i][j] = rand() % MAX_RAND; // Les nombres aléatoires ne dépasssent pas MAX_RAND.
		}
	}
}

// Initialiser une matrice à zero.
#define _zero(M) zeroMatrix(M)
float zeroMatrix(matrix M) {
	foreach(i, 0, N) {
		foreach(j, 0, N) {
			M[i][j] = 0;
		}
	}
}

// Afficher une matrice.
#define _print(M) printMatrix(M)
float printMatrix(matrix M) {
	foreach(i, 0, N) {
		printf("\n");
		foreach(j, 0, N) {
			printf("%.3f\t", M[i][j]);
		}
	}
	printf("\n");	
}


int main() {
	int length, // Juste pour récupérer le nom du processus courant.
		offset,
		remaining,
		workerId,
		workerRows, 
		workersNumber;
	char processorName[MPI_MAX_PROCESSOR_NAME]; // Une chaine de caractères qui va contenir le nom du processus courant.
	double start, end;

	matrix A, B, C; // Déclaration de matrices.
	MPI_Status status; // Déclaration du statut (status contient des informations sur le message comme: la source, tag, ...).

	_zero(C);
	// workersNumber -= 1;

	MPI_Init(NULL, NULL); // Initialiser l'environnement MPI.
	MPI_Comm_rank(MPI_COMM_WORLD, &workerId); // Récupérer l'ID du processus courant
	MPI_Comm_size(MPI_COMM_WORLD, &workersNumber); // Récupérer le nombre de processus.
	MPI_Get_processor_name(processorName, &length); // Récupérer le nom du processus courant.

	// Si le processus courant est le master.
	if (workerId == MASTER) {

		/** Le code dans ce bloc sera executer par le master seulement. **/

		start = MPI_Wtime(); // Récupérer le temps de début d'execution.

		_fill(A); // Remplir la matrice A.
		_fill(B); // Remplir la matrice B.

		_print(A); // Afficher la matrice A.
		_print(B); // Afficher la matrice B.

		/** Calculer le nombre de lignes qui seront traitées par chaque processus, 
			en divisant la taille de la matrice (N) sur le nombre de processus moins 1 (moins le processus master, car il va pas contribuer à la multiplication). **/
		workerRows = N / (workersNumber - 1);
		// Si la taille du matrice n'est pas divisible sur le nombre de processus, calculer le nombre de lignes qui restent.
		remaining = N % (workersNumber - 1);

		offset = 0; // Nous permet de garder l'indice du prochain ensemble de lignes à envoyer. Initialement on commence par 0.

		// Parcourir les processus sauf le master.
		foreach(worker, 1, workersNumber) {

			// Distribuer les lignes qui restent (en cas ou N n'est pas divisible sur (workersNumber - 1)) sur l'ensemble de processus qui ont un ID inferieur ou égale au nombre de lignes qui restent.
			if (worker <= remaining) {
				workerRows += 1;
				remaining -= 1;
				printf("remaining %d\n", remaining);
			}

			printf("Sending %d rows to worker %d with offset %d from processor %s ...\n", 
                workerRows, worker, offset, processorName);
            
            MPI_Send(&offset, 1, MPI_INT, worker, 0, MPI_COMM_WORLD); // Envoyer l'offset.
            MPI_Send(&workerRows, 1, MPI_INT, worker, 0, MPI_COMM_WORLD); // Envoyer le nombre de lignes de chaque processus.
            MPI_Send(&A[offset][0], workerRows*N, MPI_FLOAT, worker, 0, MPI_COMM_WORLD); // Envoyer à chaque processus sa partie de données de la matrice A selon l'offset et le nombre de lignes à traiter. 
            MPI_Send(&B, N*N, MPI_FLOAT, worker, 0, MPI_COMM_WORLD); // Envoyer la matrice B dans sa globalité.

            offset += workerRows; // Ajouter le nombre de lignes envoyées à l'offset pour obtenir l'indice de la séquence prochaine de lignes pour l'envoyer au processus suivant.
		}

		foreach(worker, 1, workersNumber) {
            printf("Receiving results from worker %d ...\n", worker);

            MPI_Recv(&offset, 1, MPI_INT, worker, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Recevoir l'offset pour bien placer les données dans la matrice.
            MPI_Recv(&workerRows, 1, MPI_INT, worker, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Recevoir le nombre de lignes traitées.
            MPI_Recv(&C[offset][0], workerRows*N, MPI_FLOAT, worker, 1, MPI_COMM_WORLD, &status); // Recevoir et placer le résultat de la multiplication dans C selon l'offset et le nombre de lignes traitées.
        }

        end = MPI_Wtime(); // Récupérer le temps de fin d'execution.

        _print(C);

        printf("Elapsed time: %f\n", end - start); // Calculer le temps d'execution.

	} else {
		/** Le code dans ce bloc sera executer par les autres processus. **/

		MPI_Recv(&offset, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Recevoir l'offset.
        MPI_Recv(&workerRows, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Recevoir le nombre de lignes à traiter.
        MPI_Recv(&A, workerRows*N, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status); // Recevoir la partie de données de la matrice A pour le processus courant.
        MPI_Recv(&B, N*N, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Recevoir la matrice B.

        printf("Starting multiplication by worker %d processor %s ...\n", workerId, processorName);

        // Effectuer la multiplication pour l'ensemble de données reçues.
		foreach(i, 0, workerRows) {
			foreach(j, 0, N) {
				foreach(k, 0, N) {
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}

		MPI_Send(&offset, 1, MPI_INT, MASTER, 1, MPI_COMM_WORLD); // Envoyer l'offset au master pour placer les données dans la matrice C.
        MPI_Send(&workerRows, 1, MPI_INT, MASTER, 1, MPI_COMM_WORLD); // Envoyer le nombre de lignes traitées au master.
        MPI_Send(&C, workerRows*N, MPI_FLOAT, MASTER, 1, MPI_COMM_WORLD); // Envoyer le resultat de la partie traitée.
	}

	MPI_Finalize(); // Fin de l'environnement MPI.

	return 0;
}