"""
Classes to deal with the manipulation and operations of large block diagonal
matrices and vectors in a way that:
    1. Is faster than using numpy as it doesn't store all the zeros involved.
    2. Readily works for rectangular blocks (as long as the blocks are all the
       same shape).
"""
import numpy as np

class BlockMatrix:
    def __init__(self, mat, nblock=None):
        """
        Representation of a block-diagonal matrix with off-diagonal zeros. The 
        blocks need not be square, but must have the same dimension as one
        another.

        Parameters
        ----------
        mat : 3D or 2D numpy array
            Pass either a block to repeat (2D), in which case must specify the 
            number of repeats n, or pass a list of the individual blocks to 
            repeat (3D), in which case the 2D blocks must each have the same
            shape.
        nblock : int, optional
            Specifies the number of repeats if a single block is passed. If a 
            3D mat is passed, this argument is ignored.
        """
        # Check validity of inputs
        input_mat_shape = np.shape(mat)
        mat = np.array(mat)

        if len(input_mat_shape) == 3:
            self.block_shape = np.shape(mat[0])
            for block in mat:
                if np.shape(block) != self.block_shape:
                    raise ValueError("shape of matrix blocks must be consistent.")
            nblock = input_mat_shape[0]
                
        elif len(input_mat_shape) == 2:
            if nblock is None:
                raise ValueError("must specify nblock when inputting a single matrix block.")
            self.block_shape = input_mat_shape
            mat = np.array([mat for _ in range(nblock)])
        
        else:
            raise ValueError("mat must be either 2D or 3D.")
        
        self.mat_shape = tuple( np.array(self.block_shape) * nblock )
        self.nblock = nblock
        self._matrix = mat

    def __repr__(self) -> str:
        return f"{self.block_shape} x {self.nblock} BlockMatrix"

    def __matmul__(self, other):
        """
        Return a BlockMatrix representation of the product of self and other.

        Parameters
        ----------
        other : BlockMatrix
            Must have the correct dimensionality for matrix multiplication.
        """
        # Check dimensionality is right.
        if self.mat_shape[1] != other.mat_shape[0]:
            raise ValueError("incompatible matrix shapes.")
        
        product = []
        for self_block, other_block in zip(self._matrix, other._matrix):
            product.append(self_block@other_block)
            
        if np.shape(product)[-1] == 1:
            return BlockVector(product)
        return BlockMatrix(product)
    
    def __add__(self, other):
        """
        Return a Blockmatrix representation of the sum of self and other.
        """
        # Check dimensionality is right.
        if self.mat_shape != other.mat_shape:
            raise ValueError("incompatible matrix shapes.")
        
        sum = []
        for self_block, other_block in zip(self._matrix, other._matrix):
            sum.append(self_block+other_block)
        return BlockMatrix(sum)

    def __sub__(self, other):
        """
        Return a Blockmatrix representation of the difference of self and other.
        """
        # Check dimensionality is right.
        if self.mat_shape != other.mat_shape:
            raise ValueError("incompatible matrix shapes.")
        
        difference = []
        for self_block, other_block in zip(self._matrix, other._matrix):
            difference.append(self_block-other_block)
        return BlockMatrix(sum)
    
    @property
    def matrix(self):
        """
        Return the full block-diagonal matrix.
        """
        full_mat = np.zeros(self.mat_shape)
        for n in range(self.nblock):
            b0 = self.block_shape[0]
            b1 = self.block_shape[1]
            full_mat[n*b0:(n+1)*b0, n*b1:(n+1)*b1] = self._matrix[n]
        return full_mat


class BlockVector(BlockMatrix):
    def __init__(self, vec, mode='as-is', nblock=None):
        """
        Representation of a block vector. Stores it as a series of [N,1] block
        matrices under-the-hood.

        Parameters
        ----------
        vec : 1D, 2D or 3D numpy array
            Some representation of the block diagonal vector to construct.
            Either the vector itself (1D), a single block of the vector (1D),
            all blocks of the vector in a list (2D), or the full under-the hood
            representation as a list of column vectors (3D).
        mode : str, optional
            If a 1D vec is passed, 'as-is' specifies the vector as being
            complete, while 'block' specifies it as corresponding to a single
            block of the desired vector. If a 2D vector is passed, mode is
            ignored.
        nblock : int, optional
            If a 1D vec is passed and mode is 'as-is', specifies the number of 
            blocks that the vector is composed of. If mode is 'block', specifies
            the number of desired blocks of the full vector. If a 2D vector is 
            passed, nblock is ignored.
        """
        # Check inputs and separate vector blocks.
        if len(np.shape(vec)) == 1:
            if mode == 'as-is':
                if nblock is None:
                    raise ValueError("must specify the number of blocks in the complete vector.")
                if len(vec) % nblock:
                    raise ValueError("nblock must divide length of as-is vector.")
                self.block_len = int(len(vec) / nblock)
                split_vec = [vec[n*self.block_len:(n+1)*self.block_len] for n in range(nblock)]
                mat = np.array([np.array([vector]).T for vector in split_vec])
            
            elif mode == 'block':
                if nblock is None:
                    raise ValueError("must specify the number of times to repeat vector block.")
                self.block_len = len(vec)
                one_block = np.array([vec]).T
                mat = np.array([one_block]*nblock)

            else:
                raise ValueError("invalid mode specified.")
        
        elif len(np.shape(vec)) == 2:
            nblock = len(vec)
            self.block_len = len(vec[0])
            for v in vec:
                if len(v) != self.block_len:
                    raise ValueError("length of vector blocks must be consistent.")
            mat = np.array([np.array([v]).T for v in vec])
        
        elif len(np.shape(vec)) == 3:
            if np.shape(vec)[-1] != 1:
                raise ValueError("expecting a list of column vectors. Are you inputting a list of matrices?")
            nblock = len(vec)
            self.block_len = len(vec[0])
            for v in vec:
                if len(v) != self.block_len:
                    raise ValueError("shape of vector blocks must be consistent.")
            mat = np.array(vec)
        
        else:
            raise ValueError("invalid vec shape.")

        self.nblock = nblock
        super().__init__(mat, nblock)

    def __repr__(self):
        return f"({(self.block_len)},) x {self.nblock} BlockVector"

    @property
    def vector(self):
        """
        Return the full block vector.
        """
        return self._matrix.flatten()
    
    @property
    def vector_blocks(self):
        """
        Return a list of vector blocks.
        """
        return np.split(self.vector, self.nblock)
