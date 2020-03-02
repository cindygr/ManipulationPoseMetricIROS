// Matrix Log and Exponent

template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Exp( ) const
{
   R4MatrixTC<Coord> A = (*this);           // call it A to be like Alexa's pseudocode
   R4MatrixTC<Coord> X;  X.MakeIdentity();  // the current sum
   R4MatrixTC<Coord> D;  D.MakeIdentity();  // denominator
   R4MatrixTC<Coord> N;  N.MakeIdentity();  // numerator
   Coord c = 1.0;                           // coefficienty thing

   int j = (int) max(0.0, 1.0 + floor(log(A.NormF())/log(2.0)));  // gives logbase2(A.Norm())
   A = A * (Coord)pow(2.0,-j);

   int q = 6;      // supposedly 6 is a good number of iterations
   for (int k = 1; k <= q; k++) {
      c = c*(q - k + 1.0)/(Coord)(k*(2*q - k + 1.0));
      X = A*X;
      N = N + c*X;
      D = D + (Coord)pow(-1.0,k)*c*X;
   }

   WINbool bSuc = FALSE;
   X = D.Inverse(bSuc) * N;
   int p = (int)pow(2.0,j);
   X = X.Pow(p);
   return X;
}

template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Log( /*int in_n , float &out_id*/ ) const
{
   R4MatrixTC<Coord> A = (*this);           // call it A to be like Alexa's pseudocode
   R4MatrixTC<Coord> I; I.MakeIdentity();   // identity matrix
   //A.PrintMatlab();

   int k = 0;
   int n1 = 30;
   double eps1 = 0.0001;
   while ((A-I).NormF() > eps1 /*&& k < in_n*/ ) {
      //double error = (A-I).NormF();
      A = A.Sqrt();
      k++;

      if (k > n1) {
         printf("log: repeated square roots failed to converge after %d iterations\n", n1);
         break;
      }
   }

   A = A - I;
   R4MatrixTC<Coord> Z = A;
   R4MatrixTC<Coord> X = A;
   Coord i = 1.0;
   double eps2 = 0.000000001;
   int n2 = 7;

   while ( Z.NormF() > eps2 ) {
      Z = Z*A;
      i++;
      X = X + Z/i;
      if (i > n2) {
         printf("log: failed to converge after %d iterations\n", n2);
         break;
      }
   }

   X = (Coord)pow(2.0,k)*X;
   return X;
}

template<class Coord>
inline R4MatrixTC<Coord> R4MatrixTC<Coord>::Sqrt( ) const
{
   R4MatrixTC<Coord> A = (*this);          // call it A to be like Alexa's pseudocode
   R4MatrixTC<Coord> X = (*this);
   R4MatrixTC<Coord> Y; Y.MakeIdentity();

   WINbool bSuc = FALSE;
   int i = 0;
   double eps3 = 0.000001;
   int n3 = 10;

   while ((X*X - A).NormF() > eps3) {
      double error = (X*X - A).NormF();
      R4MatrixTC<Coord> iX = X.Inverse(bSuc);
      R4MatrixTC<Coord> iY = Y.Inverse(bSuc);
      X = (X + iY)/(Coord)2.0;
      Y = (Y + iX)/(Coord)2.0;
      i++;
      if (i > n3) {
         if (error > 0.01)
             printf("sqrt: failed to converge, error = %f\n", error);
         break;
      }
   }
   return X;
}

/** \brief Take a percentage of a matrix transformation

  * E.g., for a translation matrix, translate s percentage along the way. If you took s to be 0.5, and applied the matrix you got back twice, it should be the same as applying m1 once.
  * @param s Percentage (zero gives identity matrix, one gives m1). Can use negative percentages.
  * @param m1 Matrix
  * @returns A matrix that will do the percentage transformation.
  */
template<class Coord>
inline R4MatrixTC<Coord>
ScalarMult ( Coord s, const R4MatrixTC<Coord>& m1) {
   return (s * m1.Log()).Exp();
}

/** \brief Add together two matrix transformations
  * Used in Lerp, below
  * @param m1 Matrix 1
  * @param m2 Matrix 2
  * @returns m1 + m2
  */
template<class Coord>
inline R4MatrixTC<Coord>
LinearComb (const R4MatrixTC<Coord>& m1, const R4MatrixTC<Coord>& m2) {
   return (m1.Log() + m2.Log()).Exp();
}

/** \brief Take a linear combination of two matrix transformations

  * @param s Percentage (zero gives m1, one gives m2). Can use negative percentages.
  * @param m1 Matrix
  * @param m2 Matrix
  * @returns A matrix that will do the percentage transformation.
  */
template<class Coord>
inline R4MatrixTC<Coord>
Lerp( const R4MatrixTC<Coord>& m1, const R4MatrixTC<Coord>& m2, Coord s )
{
   return LinearComb( ScalarMult(1.0-s, m1), ScalarMult(s, m2) );
}

/** \brief Take a weighted combination of n matrix transformations

  * @param weights Blend values. Should sum to one and be non-zero
  * @param mats Input matrices
  * @returns A matrix that will do the blended transformation.
  */
template<class Coord>
inline R4MatrixTC<Coord>
Blend( Array< R4MatrixTC<Coord> >& mats, const Array<double>& weights )
{
    ASSERT(mats.num() == weights.num());

    R4MatrixTC<Coord> out;
    if (weights.num() < 1) {
        out.MakeIdentity();
        return out;
    } else out = weights[0] * mats[0].Log();

    for (int i = 1; i < mats.num(); i++){
        out += ( weights[i] * mats[i].Log() );
    }
    return out.Exp();
}
