#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

struct Body {
  double x, y, m, fx, fy;
};

int main(int argc, char** argv) {
  const int N = 20;
  // 初期化
  MPI_Init(&argc, &argv);
  int size, rank; // size=ノード数，rank=自ノードの番号={0,1,2,...,size-1}
  // 決まり文句
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // 自ノードで計算する物体(=N体のうちN/size個だけ)のメモリのみを確保する．
  // ibody: 自ノード担当分の物体，jbody: 隣接ノード担当分の物体，buffer: Read/Write競合回避用の作業領域
  Body ibody[N/size], jbody[N/size], buffer[N/size];
  srand48(rank);
  for(int i=0; i<N/size; i++) {
    ibody[i].x = jbody[i].x = drand48();
    ibody[i].y = jbody[i].y = drand48();
    ibody[i].m = jbody[i].m = drand48();
    ibody[i].fx = jbody[i].fx = ibody[i].fy = jbody[i].fy = 0;
  }
  // send_to: 送信先ノード番号．自分の1つ前のノード番号を指定する
  int send_to = (rank - 1 + size) % size;
  // Body型のMPI版：MPI_BODYを定義する
  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);
  // 各プロセスのメモリ上に，ウィンドウと呼ばれる公開領域を設定
  MPI_Win win;
  // jbody変数の領域をウィンドウに設定，Body型ひとつ分を通信単位とする．
  MPI_Win_create(jbody, (N/size)*sizeof(Body), sizeof(Body), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  
  // 隣接ノードを取り替えながら，size回くりかえせば計算が完了する
  // irank回目の計算では，もともとは(irank-1)個はなれていたノードが隣接ノードになっている．
  for(int irank=0; irank < size; irank++) {
    // 隣接ノードを取り替える．
    // jbodyをbufferにコピーしてから，隣接ノードにデータを送る(=jbodyに書き込まれる)
    // jbodyを直接送るとRead/Writeが競合するので，bufferを経由させるのである．
    for(int i=0; i<N/size; i++)
      buffer[i] = jbody[i];
    MPI_Win_fence(0, win);
    MPI_Put(buffer, N/size, MPI_BODY, send_to, 0, N/size, MPI_BODY, win);
    MPI_Win_fence(0, win);
    // 隣接ノードが持っている物体とのN体計算を行う．
    // ibodyに計算結果が格納されてゆく．
    for(int i=0; i<N/size; i++) {
      for(int j=0; j<N/size; j++) {
        double rx = ibody[i].x - jbody[j].x;
        double ry = ibody[i].y - jbody[j].y;
        double r = std::sqrt(rx * rx + ry * ry);
        if (r > 1e-15) {
          ibody[i].fx -= rx * jbody[j].m / (r * r * r);
          ibody[i].fy -= ry * jbody[j].m / (r * r * r);
        }
      }
    }
  }
  // 全ノードについて，自ノードが担当する物体の計算が完了する．
  for(int irank=0; irank < size; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(irank==rank) {
      for(int i=0; i<N/size; i++) {
        printf("%d %g %g\n",i+rank*N/size,ibody[i].fx,ibody[i].fy);
      }
    }
  }
  MPI_Finalize();
}
